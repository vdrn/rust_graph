use core::time::Duration;
use std::sync::mpsc;
use web_time::Instant;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::graph_ui::plot_elements::RawPlotElements;

struct ScheduledWork {
	work:      Vec<Box<dyn FnOnce() -> ExecutionResult + Send>>,
	timestamp: Instant,
}
struct FinishedWork {
	duration:  Duration,
	timestamp: Instant,
	result:    Vec<ExecutionResult>,
}
pub type ExecutionResult = Result<(u64, RawPlotElements), (u64, String)>;
pub struct PlotElementsScheduler {
	scheduled:       bool,
	deferred:        Option<ScheduledWork>,
	average:         (Duration, usize),
	latest_received: Option<Instant>,

	sx: mpsc::SyncSender<FinishedWork>,
	rx: mpsc::Receiver<FinishedWork>,
}
impl PlotElementsScheduler {
	pub fn new() -> Self {
		let (sx, rx) = mpsc::sync_channel(4);
		Self { scheduled: false, average: (Duration::ZERO, 0), latest_received: None, deferred: None, sx, rx }
	}
	pub fn schedule(&mut self, work: Vec<Box<dyn FnOnce() -> ExecutionResult + Send + 'static>>) {
		let to_schedule = ScheduledWork { work, timestamp: Instant::now() };

		if !self.scheduled {
			self.spawn_as_latest(to_schedule);
		} else {
			self.deferred = Some(to_schedule);
		}
	}
	fn spawn_as_latest(&mut self, scheduled: ScheduledWork) {
		let sx = self.sx.clone();

		rayon::spawn(move || {
			let start = Instant::now();
			let result = scheduled
				.work
				.into_par_iter()
				.map(|work| {
					let result = work();
					result
				})
				.collect::<Vec<_>>();

			let duration = start.elapsed();

			let _send_res = sx.send(FinishedWork { duration, timestamp: scheduled.timestamp, result });
		});

		self.scheduled = true;
	}
	/// returns (has_outstanding, maybe_new_result)
	pub fn try_receive(&mut self) -> (bool, Option<Vec<ExecutionResult>>) {
		// let mut ido = None;

		let mut result = None;
		while let Ok(finished_work) = self.rx.try_recv() {
			self.scheduled = false;
			self.average.1 += 1;
			self.average.0 += finished_work.duration;

			if let Some(latest) = self.latest_received {
				if finished_work.timestamp < latest {
					continue;
				}
			}
			self.latest_received = Some(finished_work.timestamp);
			result = Some(finished_work.result);
		}

		let has_outstanding = self.scheduled;
		(has_outstanding, result)
	}

	pub fn schedule_deffered_if_idle(&mut self) -> bool {
		if !self.scheduled {
			if let Some(work) = self.deferred.take() {
				self.spawn_as_latest(work);
				return true;
			}
		}
		false
	}
}
