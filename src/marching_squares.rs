use crate::scope;
use ahash::AHashMap;
use arrayvec::ArrayVec;
use egui_plot::PlotPoint;
use smallvec::SmallVec;

pub fn marching_squares(
	f: impl Fn(f64, f64) -> Result<f64, String>, bounds_min: (f64, f64), bounds_max: (f64, f64),
	resolution: usize,
) -> Result<Vec<Vec<PlotPoint>>, String> {
	scope!("marching_squares");
	let (x_min, y_min) = bounds_min;
	let (x_max, y_max) = bounds_max;

	let dx = (x_max - x_min) / resolution as f64;
	let dy = (y_max - y_min) / resolution as f64;

	let mut grid = vec![vec![0.0; resolution + 1]; resolution + 1];
	{
		scope!("grid_calc");

		for i in 0..=resolution {
			for j in 0..=resolution {
				let x = x_min + i as f64 * dx;
				let y = y_min + j as f64 * dy;
				grid[i][j] = f(x, y)?;
			}
		}
	}

	let mut polyline_builder = PolylineBuilder::new(0.0001, f32::EPSILON as f64);
	// let mut polylines: Vec<Vec<PlotPoint>> = Vec::new();

	scope!("generate_polylines");
	for i in 0..resolution {
		for j in 0..resolution {
			// scope!("grid_cell");
			let x = x_min + i as f64 * dx;
			let y = y_min + j as f64 * dy;

			// vals at (bottom-left, bottom-right, top-right, top-left)
			let vals = [grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]];

			// 1 if positive, 0 if negative
			let mut config = 0u8;
			for (idx, &val) in vals.iter().enumerate() {
				if val > 0.0 {
					config |= 1 << idx;
				}
			}

			for (edge1, edge2) in get_edges_for_config(config) {
				let p1 = interpolate_edge(edge1, x, y, dx, dy, &vals);
				let p2 = interpolate_edge(edge2, x, y, dx, dy, &vals);

				polyline_builder.add_segment(PlotPoint::new(p1.0, p1.1), PlotPoint::new(p2.0, p2.1));
				// add_segment_to_polylines(
				// 	&mut polylines,
				// 	PlotPoint::new(p1.0, p1.1),
				// 	PlotPoint::new(p2.0, p2.1),
				// 	eps,
				// );
			}
		}
	}

	let polylines = polyline_builder.finish();
	// println!("polylines: {}", polylines.len());
	Ok(polylines)
}
/// Get the edges that the contour crosses for a given configuration
/// Returns pairs of edges: (entry_edge, exit_edge)
/// 0 - bottom, 1 - right, 2 - top, 3 - left
fn get_edges_for_config(config: u8) -> ArrayVec<(Edge, Edge), 2> {
	let mut result = ArrayVec::new();
	match config {
		0 | 15 => {}, // All same sign
		1 | 14 => result.push((Edge::Bottom, Edge::Left)),
		2 | 13 => result.push((Edge::Bottom, Edge::Right)),
		3 | 12 => result.push((Edge::Right, Edge::Left)),
		4 | 11 => result.push((Edge::Right, Edge::Top)),
		5 => {
			// ambiguous
			result.push((Edge::Bottom, Edge::Right));
			result.push((Edge::Top, Edge::Left));
		},
		6 | 9 => result.push((Edge::Bottom, Edge::Top)),
		7 | 8 => result.push((Edge::Top, Edge::Left)),
		10 => {
			// ambiguous
			result.push((Edge::Bottom, Edge::Left));
			result.push((Edge::Right, Edge::Top));
		},
		_ => {
			unreachable!();
		},
	}
	result
}
enum Edge {
	Bottom,
	Right,
	Top,
	Left,
}

/// Interpolate position on edge where the function crosses 0
/// 0 - bottom, 1 - right, 2 - top, 3 - left
fn interpolate_edge(edge: Edge, x: f64, y: f64, dx: f64, dy: f64, vals: &[f64; 4]) -> (f64, f64) {
	match edge {
		Edge::Bottom => {
			// corners 0 and 1
			let t = vals[0] / (vals[0] - vals[1]);
			(x + t * dx, y)
		},
		Edge::Right => {
			// corners 1 and 2
			let t = vals[1] / (vals[1] - vals[2]);
			(x + dx, y + t * dy)
		},
		Edge::Top => {
			// corners 2 and 3
			let t = vals[2] / (vals[2] - vals[3]);
			(x + (1.0 - t) * dx, y + dy)
		},
		Edge::Left => {
			// corners 3 and 0
			let t = vals[3] / (vals[3] - vals[0]);
			(x, y + (1.0 - t) * dy)
		},
	}
}

// fn add_segment_to_polylines(polylines: &mut Vec<Vec<PlotPoint>>, p1: PlotPoint, p2: PlotPoint, eps: f64) {
// 	// scope!("add_segment_to_polylines");
// 	// Find polylines that this segment connects to
// 	let mut matches: ArrayVec<(usize, ConnectType), 2> = ArrayVec::new();

// 	for (idx, polyline) in polylines.iter().enumerate() {
// 		let start = polyline.first().unwrap();
// 		let end = polyline.last().unwrap();

// 		if equals(end, &p1, eps) {
// 			matches.push((idx, ConnectType::EndToP1));
// 		} else if equals(end, &p2, eps) {
// 			matches.push((idx, ConnectType::EndToP2));
// 		} else if equals(start, &p1, eps) {
// 			matches.push((idx, ConnectType::StartToP1));
// 		} else if equals(start, &p2, eps) {
// 			matches.push((idx, ConnectType::StartToP2));
// 		}
// 		if matches.len() == 2 {
// 			break;
// 		}
// 	}

// 	match matches.len() {
// 		0 => {
// 			polylines.push(vec![p1, p2]);
// 		},
// 		1 => {
// 			// append to existing polyline
// 			let (idx, connect_type) = matches[0];
// 			match connect_type {
// 				ConnectType::EndToP1 => polylines[idx].push(p2),
// 				ConnectType::EndToP2 => polylines[idx].push(p1),
// 				ConnectType::StartToP1 => polylines[idx].insert(0, p2),
// 				ConnectType::StartToP2 => polylines[idx].insert(0, p1),
// 			}
// 		},
// 		2 => {
// 			// 2 matches - either close a loop or merge them
// 			let (idx1, type1) = matches[0];
// 			let (idx2, type2) = matches[1];

// 			if idx1 == idx2 {
// 				// the polyline connected to itself - nothing more to do
// 			} else {
// 				// scope!("merge_polylines");

// 				// Merge two polylines
// 				let polyline2 = polylines.remove(idx2.max(idx1));
// 				let polyline1 = polylines.remove(idx1.min(idx2));

// 				// Determine correct merge order based on connection types
// 				let merged = merge_polylines(polyline1, polyline2, type1, type2);
// 				polylines.push(merged);
// 			}
// 		},
// 		_ => {
// 			unreachable!();
// 			// polylines.push(vec![p1, p2]);
// 		},
// 	}
// }

// #[derive(Debug, Clone, Copy)]
// enum ConnectType {
// 	EndToP1,
// 	EndToP2,
// 	StartToP1,
// 	StartToP2,
// }

// fn merge_polylines(
// 	mut p1: Vec<PlotPoint>, mut p2: Vec<PlotPoint>, type1: ConnectType, type2: ConnectType,
// ) -> Vec<PlotPoint> {
// 	use ConnectType as CT;

// 	match (type1, type2) {
// 		(CT::EndToP1, CT::StartToP2) | (CT::EndToP2, CT::StartToP1) => {
// 			// p1.end connects to segment, p2.start connects to segment
// 			p1.extend(p2);
// 			p1
// 		},
// 		(CT::EndToP1, CT::EndToP2) | (CT::EndToP2, CT::EndToP1) => {
// 			// Both ends connect - reverse p2
// 			p2.reverse();
// 			p1.extend(p2);
// 			p1
// 		},
// 		(CT::StartToP1, CT::StartToP2) | (CT::StartToP2, CT::StartToP1) => {
// 			// Both starts connect - reverse p1
// 			p1.reverse();
// 			p1.extend(p2);
// 			p1
// 		},
// 		(CT::StartToP1, CT::EndToP2) | (CT::StartToP2, CT::EndToP1) => {
// 			// p2.end connects to segment, p1.start connects to segment
// 			p2.extend(p1);
// 			p2
// 		},
// 		_ => {
// 			// Shouldn't happen, but just concatenate
// 			p1.extend(p2);
// 			p1
// 		},
// 	}
// }

fn equals(p1: &PlotPoint, p2: &PlotPoint, eps: f64) -> bool {
	(p1.x - p2.x).abs() < eps && (p1.y - p2.y).abs() < eps
}

#[derive(Clone, Copy)]
struct HashablePoint {
	x: i64,
	y: i64,
}

impl HashablePoint {
	fn from_point(p: &PlotPoint, precision_recip: f64) -> Self {
		// quantize to grid
		HashablePoint { x: (p.x * precision_recip).round() as i64, y: (p.y * precision_recip).round() as i64 }
	}
}

impl PartialEq for HashablePoint {
	fn eq(&self, other: &Self) -> bool { self.x == other.x && self.y == other.y }
}

impl Eq for HashablePoint {}

impl core::hash::Hash for HashablePoint {
	fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
		self.x.hash(state);
		self.y.hash(state);
	}
}

struct PolylineBuilder {
	polylines:       Vec<Vec<PlotPoint>>,
	/// (polyline index, is_start)
	endpoint_map:    AHashMap<HashablePoint, SmallVec<[(u32, bool); 2]>>,
	eps:       f64,
	precision_recip: f64,
}

impl PolylineBuilder {
	fn new(grid_precision: f64, eps:f64) -> Self {
		PolylineBuilder {
			polylines: Vec::new(),
			endpoint_map: AHashMap::new(),
			eps,
			precision_recip: 1.0 / grid_precision,
		}
	}

	fn add_segment(&mut self, start: PlotPoint, end: PlotPoint) {
		let start_hash = HashablePoint::from_point(&start, self.precision_recip);
		let end_hash = HashablePoint::from_point(&end, self.precision_recip);

		let mut connects_at_start: Option<(usize, bool)> = None;
		let mut connects_at_end: Option<(usize, bool)> = None;

		// Check if segment.start connects to any polyline endpoint
		if let Some(candidates) = self.endpoint_map.get(&start_hash) {
			for &(idx, is_start) in candidates {
				let idx = idx as usize;
				if idx >= self.polylines.len() {
					// Skip stale indices
					continue;
				}
				let endpoint =
					if is_start { &self.polylines[idx][0] } else { self.polylines[idx].last().unwrap() };

				if equals(endpoint, &start, self.eps) {
					connects_at_start = Some((idx, is_start));
					break;
				}
			}
		}

		// Check if segment.end connects to any polyline endpoint
		if let Some(candidates) = self.endpoint_map.get(&end_hash) {
			for &(idx, is_start) in candidates {
				let idx = idx as usize;
				if idx >= self.polylines.len() {
					// Skip stale indices
					continue;
				}
				let endpoint =
					if is_start { &self.polylines[idx][0] } else { self.polylines[idx].last().unwrap() };

				if equals(endpoint, &end, self.eps) {
					connects_at_end = Some((idx, is_start));
					break;
				}
			}
		}

		match (connects_at_start, connects_at_end) {
			// Segment connects two different polylines
			(Some((idx1, at_start1)), Some((idx2, at_start2))) if idx1 != idx2 => {
				self.merge_polylines(idx1, at_start1, idx2, at_start2);
			},

			// Segment closes a loop
			(Some((idx, _)), Some((_, _))) => {
				self.remove_endpoint(&start, idx, false);
				self.remove_endpoint(&end, idx, true);
			},

			// Segment extends polyline at its end
			(Some((idx, false)), None) => {
				self.remove_endpoint(&start, idx, false);
				self.polylines[idx].push(end);
				self.add_endpoint(&end, idx, false);
			},

			// Segment extends polyline at its start
			(Some((idx, true)), None) => {
				self.remove_endpoint(&start, idx, true);
				self.polylines[idx].insert(0, end);
				self.add_endpoint(&end, idx, true);
			},

			// Segment prepends to polyline start
			(None, Some((idx, true))) => {
				self.remove_endpoint(&end, idx, true);
				self.polylines[idx].insert(0, start);
				self.add_endpoint(&start, idx, true);
			},

			// Segment appends to polyline end
			(None, Some((idx, false))) => {
				self.remove_endpoint(&end, idx, false);
				self.polylines[idx].push(start);
				self.add_endpoint(&start, idx, false);
			},

			// New disconnected polyline
			(None, None) => {
				let idx = self.polylines.len();
				self.polylines.push(vec![start, end]);
				self.add_endpoint(&start, idx, true);
				self.add_endpoint(&end, idx, false);
			},
		}
	}

	fn add_endpoint(&mut self, point: &PlotPoint, polyline_idx: usize, is_start: bool) {
		let hash = HashablePoint::from_point(point, self.precision_recip);
		self.endpoint_map.entry(hash).or_default().push((polyline_idx as u32, is_start));
	}

	fn remove_endpoint(&mut self, point: &PlotPoint, polyline_idx: usize, is_start: bool) {
		let hash = HashablePoint::from_point(point, self.precision_recip);
		if let Some(entries) = self.endpoint_map.get_mut(&hash) {
			entries.retain(|&mut (idx, start)| idx as usize != polyline_idx || start != is_start);
			if entries.is_empty() {
				self.endpoint_map.remove(&hash);
			}
		}
	}

	fn merge_polylines(&mut self, idx1: usize, at_start1: bool, idx2: usize, at_start2: bool) {
		let p1 = if at_start1 { self.polylines[idx1][0] } else { *self.polylines[idx1].last().unwrap() };
		let p2 = if at_start2 { self.polylines[idx2][0] } else { *self.polylines[idx2].last().unwrap() };

		self.remove_endpoint(&p1, idx1, at_start1);
		self.remove_endpoint(&p2, idx2, at_start2);

		let (keep_idx, remove_idx) = if idx1 < idx2 { (idx1, idx2) } else { (idx2, idx1) };

		let mut removed = self.polylines.swap_remove(remove_idx);

		// Update indices
		if remove_idx < self.polylines.len() {
			let swapped_from_idx = self.polylines.len();
			self.update_polyline_index(swapped_from_idx, remove_idx);
		}

		let mut keep = core::mem::take(&mut self.polylines[keep_idx]);

		// Determine which configuration we have:
		// - idx1 end connects to idx2 start: append idx2 to idx1
		// - idx1 start connects to idx2 end: prepend idx1 to idx2
		// - idx1 end connects to idx2 end: reverse idx2, then append
		// - idx1 start connects to idx2 start: reverse idx2, then prepend

		if keep_idx == idx1 {
			// keep idx1
			match (at_start1, at_start2) {
				(false, true) => {
					// idx1.end to idx2.start: append idx2 to idx1
					keep.extend(removed);
				},
				(false, false) => {
					// idx1.end to idx2.end: reverse idx2, then append
					removed.reverse();
					keep.extend(removed);
				},
				(true, true) => {
					// idx1.start to idx2.start: reverse idx2, then prepend
					removed.reverse();
					removed.extend(keep);
					keep = removed;
				},
				(true, false) => {
					// idx1.start to idx2.end: prepend idx2 to idx1
					removed.extend(keep);
					keep = removed;
				},
			}
		} else {
			// keep idx2
			match (at_start1, at_start2) {
				(false, true) => {
					// idx1.end to idx2.start: prepend idx1 to idx2
					removed.extend(keep);
					keep = removed;
				},
				(false, false) => {
					// idx1.end to idx2.end: reverse idx1, then append to idx2
					removed.reverse();
					keep.extend(removed);
				},
				(true, true) => {
					// idx1.start to idx2.start: reverse idx1, then prepend to idx2
					removed.reverse();
					removed.extend(keep);
					keep = removed;
				},
				(true, false) => {
					// idx1.start to idx2.end: append idx1 to idx2
					keep.extend(removed);
				},
			}
		}

		self.polylines[keep_idx] = keep;

		let new_start = self.polylines[keep_idx][0];
		let new_end = *self.polylines[keep_idx].last().unwrap();
		self.add_endpoint(&new_start, keep_idx, true);
		self.add_endpoint(&new_end, keep_idx, false);
	}
	fn update_polyline_index(&mut self, old_idx: usize, new_idx: usize) {
		for entries in self.endpoint_map.values_mut() {
			for entry in entries {
				if entry.0 as usize == old_idx {
					entry.0 = new_idx as u32;
				}
			}
		}
	}

	fn finish(self) -> Vec<Vec<PlotPoint>> { self.polylines }
}
