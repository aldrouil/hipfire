// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! A reusable GPU span timer: one lazily created start/stop `hipEvent` pair per
//! [`Gpu`], recorded around a span of launches on that `Gpu`'s own stream and
//! read once the span's own stream-ordered completion has been observed.
//!
//! A caller that has to attribute one GPU span per step — the pass-back split's
//! GPU arm is the one such caller today — would otherwise create and destroy two
//! events per step. Here the pair is created once per device and reused; neither
//! recording it nor reading it synchronizes anything. The records are
//! stream-ordered markers, and [`Gpu::span_elapsed_ns`] is only meaningful once
//! the caller has observed the span complete by other means: the pass-back reads
//! it after the step's *completed* H2D, which is ordered behind both markers on
//! the same stream, so the measurement costs no event synchronization and no
//! extra fence.
//!
//! ## Why the pair is never re-recorded while in flight
//!
//! HIP leaves re-recording an event whose previous record has not completed
//! undefined, so the pair is never recorded into again while a previous span may
//! still be outstanding. [`SpanTimer`] tracks how far the span has got in one
//! explicit state ([`SpanState`]), which only a *successful* HIP call advances
//! and which otherwise drops to `Failed` *before* each record is attempted — a
//! record that reports a failure may still have submitted the marker, so
//! "attempted" is the only state that is safe to reuse or destroy from.
//! [`SpanTimer::begin`] therefore drains the owning device
//! (`hipDeviceSynchronize`) before it records into a pair whose previous span was
//! never resolved; [`SpanTimer::end`] and [`Gpu::span_elapsed_ns`] refuse a span
//! that is not open or not complete instead of re-recording or reading it.
//!
//! Every failure here is reported to the caller, which fails its step: the pair
//! is a measurement, and a measurement that cannot be taken must not silently
//! leave a caller with a stale or missing one. On the normal path no failure
//! happens: the previous span was resolved by [`Gpu::span_elapsed_ns`], so
//! reuse adds no host wait, and an unresolved pair is drained exactly once more,
//! at teardown, before its handles are destroyed — retained instead if even that
//! drain fails.

use hip_bridge::{Event, HipError, HipResult, HipRuntime, Stream};

use crate::Gpu;

/// The start/stop `hipEvent` pair of one GPU span, and how far that span has got.
///
/// One explicit [`SpanState`], not two flags, is what makes reuse safe *and* what
/// keeps a half-finished span out of a read. The state is advanced only by a
/// *successful* HIP call, except that each record drops it to
/// [`SpanState::Failed`] *before* the attempt: a record that reports a failure
/// may still have submitted its marker, so "attempted" is the only state safe to
/// reuse or destroy from. Keeping that state *distinct* from the open and
/// complete ones is what stops a caller that ignored an error from reading a
/// previous span's markers — after a failed `begin` only `Failed` is set, and
/// after a failed `end` the state is `Failed`, not `Recorded`, so `end` and
/// `elapsed` refuse instead of re-recording or reading the earlier record.
pub(crate) struct SpanTimer {
    start: Event,
    stop: Event,
    state: SpanState,
}

/// How far the span whose markers [`SpanTimer`] holds has got.
///
/// `Idle` is the only state the pair may be reused or destroyed from without a
/// drain; in every other state a marker may be in flight, so [`SpanTimer::settle`]
/// (and teardown) synchronizes the owning device first. `Started` and `Recorded`
/// are what a *successful* record advances to; `Failed` is where a record attempt
/// that did not succeed leaves it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SpanState {
    /// No marker in flight; the pair may be reused without draining.
    Idle,
    /// The start marker was recorded: the span is open.
    Started,
    /// Both markers were recorded: the span may be read.
    Recorded,
    /// A record was attempted and did not succeed, so a marker may be in flight:
    /// the pair must be drained before it is reused.
    Failed,
}

impl SpanTimer {
    /// Create the pair. A failure on the second event destroys the first, so a
    /// partial allocation cannot leak a handle; a cleanup that itself fails is
    /// reported before the create error is returned, not swallowed.
    fn create(hip: &HipRuntime) -> HipResult<Self> {
        let start = hip.event_create()?;
        let stop = match hip.event_create() {
            Ok(stop) => stop,
            Err(error) => {
                if let Err(cleanup) = hip.event_destroy(start) {
                    eprintln!(
                        "[rdna-compute] failed to destroy the start event of a partially \
                         created GPU span timer: {cleanup}"
                    );
                }
                return Err(error);
            }
        };
        Ok(SpanTimer {
            start,
            stop,
            state: SpanState::Idle,
        })
    }

    /// Record the span's start marker on `stream`, first settling any previous
    /// span so a marker that may still be in flight is not recorded over.
    fn begin(&mut self, hip: &HipRuntime, stream: Option<&Stream>) -> HipResult<()> {
        self.settle(hip)?;
        // Before the attempt, not after: see the struct docs.
        self.state = SpanState::Failed;
        hip.event_record(&self.start, stream)?;
        self.state = SpanState::Started;
        Ok(())
    }

    /// Record the span's end marker on `stream`.
    ///
    /// Refuses a span that is not open: one that was never begun, one already
    /// bracketed, and one left `Failed` by an earlier record (whose marker may
    /// still be in flight) would each re-record a marker that must not be
    /// re-recorded.
    ///
    /// A record that fails leaves the state `Failed` (it is dropped before the
    /// attempt, see the struct docs), so the span is still refused a second stop
    /// and still drained before any reuse — and [`SpanTimer::elapsed_ns`] refuses
    /// it too, so a caller that ignored the failure cannot read the pair against a
    /// previous stop. The pass-back propagates the failure all the same.
    fn end(&mut self, hip: &HipRuntime, stream: Option<&Stream>) -> HipResult<()> {
        match self.state {
            SpanState::Started => {}
            SpanState::Idle => {
                return Err(HipError::new(0, "GPU span end without a recorded begin"));
            }
            SpanState::Recorded => {
                return Err(HipError::new(
                    0,
                    "GPU span end with the span already bracketed",
                ));
            }
            SpanState::Failed => {
                return Err(HipError::new(
                    0,
                    "GPU span end with the span unresolved after a failed record",
                ));
            }
        }
        // Before the attempt, not after: see the struct docs.
        self.state = SpanState::Failed;
        hip.event_record(&self.stop, stream)?;
        self.state = SpanState::Recorded;
        Ok(())
    }

    /// Nanoseconds elapsed between the completed markers. Never synchronizes:
    /// the caller reads this only after the span's own stream-ordered
    /// completion has been observed, so both markers are already complete.
    ///
    /// Only a `Recorded` span is read: any other state either never bracketed the
    /// pair or may have a marker in flight, and the elapsed time of markers from
    /// two different spans would be neither one's.
    ///
    /// The state stays `Recorded` until the read *succeeds*: a failed
    /// `hipEventElapsedTime` (`hipErrorNotReady` on a marker that has not
    /// completed, or a runtime error) preserves it, so the pair is still refused a
    /// re-record and is drained before reuse or destruction — a read failure must
    /// never be the reason a live marker is recorded over. Only a successful read
    /// returns the pair to `Idle`.
    fn elapsed_ns(&mut self, hip: &HipRuntime) -> HipResult<u64> {
        match self.state {
            SpanState::Recorded => {}
            SpanState::Idle => {
                return Err(HipError::new(
                    0,
                    "GPU span elapsed without a recorded begin",
                ));
            }
            SpanState::Started => {
                return Err(HipError::new(0, "GPU span elapsed without a recorded end"));
            }
            SpanState::Failed => {
                return Err(HipError::new(
                    0,
                    "GPU span elapsed with the span unresolved after a failed record",
                ));
            }
        }
        let ms = hip.event_elapsed_ms(&self.start, &self.stop)?;
        // Resolved only now, after the read succeeded: both markers completed for
        // the call to have returned at all.
        self.state = SpanState::Idle;
        let ns = (ms as f64 * 1e6) as u64;
        if ns == 0 {
            // Below the event timer's resolution. Every real span this mechanism
            // is used for is tens of microseconds at the very least, so a zero
            // reading is a broken measurement, not a very fast arm — and handing
            // it to a rate would read as an infinite one. The markers themselves
            // did complete (the call above succeeded), so the span stays
            // resolved: only the value is rejected.
            return Err(HipError::new(0, "GPU span elapsed below timer resolution"));
        }
        Ok(ns)
    }

    /// Drain any non-`Idle` span so the handles are quiescent and the pair can be
    /// recorded into again.
    fn settle(&mut self, hip: &HipRuntime) -> HipResult<()> {
        if self.state == SpanState::Idle {
            return Ok(());
        }
        // The recorded markers' stream may have been destroyed since (a failed
        // step or a stream switch), so drain the owning device rather than a
        // possibly stale handle — the same rule `PinnedTransferBuffer::begin`
        // follows for the same transition.
        hip.device_synchronize()?;
        self.state = SpanState::Idle;
        Ok(())
    }

    /// Destroy both handles. The caller must have drained an unresolved span.
    fn destroy(self, hip: &HipRuntime) -> HipResult<()> {
        let start = hip.event_destroy(self.start);
        let stop = hip.event_destroy(self.stop);
        start.and(stop)
    }

    /// Teardown: drain an unresolved span, then destroy the pair.
    ///
    /// A failed drain retains both handles (they are dropped, not destroyed):
    /// an event whose record may still be in flight must not be destroyed, and
    /// leaking one at teardown is the safe end of that trade — the same
    /// retention `PinnedTransferBuffer`'s destructor makes on a failed drain.
    pub(crate) fn teardown(self, hip: &HipRuntime) {
        if self.state != SpanState::Idle {
            if let Err(err) = hip.device_synchronize() {
                eprintln!(
                    "[rdna-compute] failed to synchronize an unresolved GPU span during Gpu \
                     drop: {err} — retaining the event handles"
                );
                return;
            }
        }
        if let Err(err) = self.destroy(hip) {
            eprintln!("[rdna-compute] failed to destroy the GPU span timer during Gpu drop: {err}");
        }
    }
}

impl Gpu {
    /// Record the start marker of a GPU span on this device's active stream,
    /// creating the reusable pair on first use.
    ///
    /// Nothing is synchronized on the normal path: the pair is only recorded
    /// into, and the caller reads it with [`Gpu::span_elapsed_ns`] once the
    /// span's own stream-ordered completion has been observed. A marker that
    /// cannot be created or recorded is returned to the caller, which fails its
    /// step: the pair is a measurement, and the pass-back does not run a step
    /// whose arms it could not measure.
    pub fn span_begin(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        if self.span_timer.is_none() {
            self.span_timer = Some(SpanTimer::create(&self.hip)?);
        }
        let timer = self.span_timer.as_mut().expect("created above");
        timer.begin(&self.hip, self.active_stream.as_ref())
    }

    /// Record the end marker of the span [`Gpu::span_begin`] started, on the
    /// same stream. Errors when no span is open (a failed `span_begin`) or when
    /// one is already bracketed.
    pub fn span_end(&mut self) -> HipResult<()> {
        // Bind rather than assume this thread's last GPU call was this device's:
        // the marker goes to *this* device's stream, and a wrong active device is
        // an error the caller must see, not a marker recorded somewhere else.
        self.bind_thread()?;
        let timer = self
            .span_timer
            .as_mut()
            .ok_or_else(|| HipError::new(0, "GPU span end with no span timer"))?;
        timer.end(&self.hip, self.active_stream.as_ref())
    }

    /// Nanoseconds of the last bracketed span. Synchronizes nothing; see
    /// [`Gpu::span_begin`].
    pub fn span_elapsed_ns(&mut self) -> HipResult<u64> {
        self.bind_thread()?;
        let timer = self
            .span_timer
            .as_mut()
            .ok_or_else(|| HipError::new(0, "GPU span elapsed with no span timer"))?;
        timer.elapsed_ns(&self.hip)
    }
}
