use eframe::egui::{Color32, Id, Mesh, Pos2, Vec2};
use egui_plot::{LineStyle, PlotBounds, PlotPoint, PlotTransform};

pub struct DrawLine2 {
	pub id:            Id,
	pub name:          String,
	pub sorting_index: u32,
	pub allow_hover:   bool,
	pub color:         Color32,
	pub series:        Vec<(PlotPoint, Color32)>,
	pub style:         LineStyle,
	pub width:         f32,
}
impl DrawLine2 {
	pub fn new(
		sorting_index: u32, id: Id, name: String, allow_hower: bool, width: f32, style: LineStyle,
		base_color: Color32, series: Vec<(PlotPoint, Color32)>,
	) -> Self {
		Self { sorting_index, id, name, allow_hover: allow_hower, width, style, series, color: base_color }
	}
	pub fn points(&self) -> &Vec<(PlotPoint, Color32)> { &self.series }
}

pub struct Tessellator {
	feathering: f32,
}
impl Tessellator {
	pub fn new(pixels_per_point: f32) -> Self {
		let mut new = Self { feathering: 1.0 };
    new.set_pixels_per_point(pixels_per_point);
    new

	}
  pub fn set_pixels_per_point(&mut self, pixels_per_point: f32) {
    self.feathering = 1.0 / pixels_per_point;
  }
	pub fn tessalate_line(
		&self, line: &DrawLine2, transform: &PlotTransform, out: &mut Mesh,
	) -> PlotBounds {
		let mut bounds = PlotBounds::NOTHING;
		match line.series.len() {
			0 => {},
			1 => {
				let radius = line.width * 0.5;
				// todo circle
				bounds.extend_with(&line.series[0].0);
				let pt = transform.position_from_point(&line.series[0].0);
				Self::tesselate_circle(pt, radius, line.series[0].1, out);
			},
			_ => {
				match line.style {
					LineStyle::Solid => {
						let new_bounds = self.tesselate_path(
							transform,
							LinePoints::Untransformed(&line.series),
							line.width,
							out,
						);
						bounds.merge(&new_bounds.unwrap());
					},
					LineStyle::Dotted { spacing } => {
						let radius = line.width;

						let mut position_on_segment = 0.0;

						bounds.extend_with(&line.series[0].0);
						for window in line.series.windows(2) {
							let (start, end) = (window[0].0, window[1].0);
							bounds.extend_with(&end);
							let (start_color, end_color) = (window[0].1, window[1].1);

							let start = transform.position_from_point(&start);
							let end = transform.position_from_point(&end);

							let vector = end - start;
							let segment_length = vector.length();
							while position_on_segment < segment_length {
								let t = position_on_segment / segment_length;
								let new_point = start + vector * t;
								let new_color = start_color.lerp_to_gamma(end_color, t);
								Self::tesselate_circle(new_point, radius, new_color, out);
								// shapes.push(Shape::circle_filled(new_point, radius, color));
								position_on_segment += spacing;
							}
							position_on_segment -= segment_length;
						}
					},
					LineStyle::Dashed { length } => {
						let golden_ratio = (5.0_f32.sqrt() - 1.0) / 2.0; // 0.61803398875
						let dash_length = length;
						let gap_length = dash_length * golden_ratio;
						let dash_offset = 0.0;

						let mut position_on_segment = dash_offset;
						let mut drawing_dash = false;
						bounds.extend_with(&line.series[0].0);
						for window in line.series.windows(2) {
							let (start, end) = (window[0].0, window[1].0);
							bounds.extend_with(&end);

							let start = transform.position_from_point(&start);
							let end = transform.position_from_point(&end);

							let (start_color, end_color) = (window[0].1, window[1].1);

							let vector = end - start;
							let segment_length = vector.length();

							let mut start_point = start;
							while position_on_segment < segment_length {
								let t = position_on_segment / segment_length;
								let new_point = start + vector * t;
								let new_color = start_color.lerp_to_gamma(end_color, t);
								if drawing_dash {
									// This is the end point.
									self.tesselate_path(
										transform,
										LinePoints::Transformed(&[
											(start_point, start_color),
											(new_point, new_color),
										]),
										line.width,
										out,
									);
									position_on_segment += gap_length;
								} else {
									// Start a new dash.
									start_point = new_point;
									position_on_segment += dash_length;
								}
								drawing_dash = !drawing_dash;
							}

							// If the segment ends and the dash is not finished, add the segment's end point.
							if drawing_dash {
								self.tesselate_path(
									transform,
									LinePoints::Transformed(&[(start_point, start_color), (end, end_color)]),
									line.width,
									out,
								);
								// shapes.push(Shape::line_segment([start_point, end], stroke));
							}

							position_on_segment -= segment_length;
						}
					},
				}
			},
		}

		bounds
	}

	fn tesselate_path(
		&self, t: &PlotTransform, line: LinePoints, width: f32, out: &mut Mesh,
	) -> Option<PlotBounds> {
		let idx = out.vertices.len() as u32;
		let n = line.len();
		if n < 2 || width <= 0.0 {
			return Some(PlotBounds::NOTHING);
		}
		let radius = width * 0.5;

		if self.feathering <= 0.0 {
			let (bounds, total_pts) = line.line_screen_points(t, |_i, pt, normal, color, _is_last| {
				out.colored_vertex(pt + normal * radius, color);
				out.colored_vertex(pt - normal * radius, color);
			});
			let total_pts = total_pts as u32;
			out.reserve_triangles(2 * total_pts as usize);
			for i in 0..total_pts - 1 {
				out.add_triangle(
					idx + (2 * i) % (2 * total_pts),
					idx + (2 * i + 1) % (2 * total_pts),
					idx + (2 * i + 2) % (2 * total_pts),
				);
				out.add_triangle(
					idx + (2 * i + 2) % (2 * total_pts),
					idx + (2 * i + 1) % (2 * total_pts),
					idx + (2 * i + 3) % (2 * total_pts),
				);
			}
			bounds
		} else {
			let color_outer = Color32::TRANSPARENT;

			// We add a bit of an epsilon here, because when we round to pixels,
			// we can get rounding errors (unless pixels_per_point is an integer).
			// And it's better to err on the side of the nicer rendering with line caps
			// (the thin-line optimization has no line caps).
			let thin_line = width <= 0.9 * self.feathering;
			if thin_line {
				// If the stroke is painted smaller than the pixel width (=feathering width),
				// then we risk severe aliasing.
				// Instead, we paint the stroke as a triangular ridge, two feather-widths wide,
				// and lessen the opacity of the middle part instead of making it thinner.

				// TODO(emilk): add line caps (if this is an open line).
				let opacity = width / self.feathering;

				/*
				We paint the line using three edges: outer, middle, fill.
				.       o   m   i      outer, middle, fill
				.       |---|          feathering (pixel width)
				*/

				let mut i0 = 0;
				let (bounds, _total_pts) = line.line_screen_points(t, |i1, p, n, color, _is_last| {
					let i1 = i1 as u32;
					let connect_with_previous = i1 > 0;
					out.colored_vertex(p + n * self.feathering, color_outer);
					out.colored_vertex(p, color.gamma_multiply(opacity));
					out.colored_vertex(p - n * self.feathering, color_outer);
					if connect_with_previous {
						out.add_triangle(idx + 3 * i0, idx + 3 * i0 + 1, idx + 3 * i1);
						out.add_triangle(idx + 3 * i0 + 1, idx + 3 * i1, idx + 3 * i1 + 1);

						out.add_triangle(idx + 3 * i0 + 1, idx + 3 * i0 + 2, idx + 3 * i1 + 1);
						out.add_triangle(idx + 3 * i0 + 2, idx + 3 * i1 + 1, idx + 3 * i1 + 2);
					}
					i0 = i1;
				});

				bounds
			} else {
				// thick anti-aliased line

				/*
				We paint the line using four edges: outer, middle, middle, fill

				.       o   m     p    m   f   outer, middle, point, middle, fill
				.       |---|                  feathering (pixel width)
				.         |--------------|     width
				.       |---------|            outer_rad
				.           |-----|            inner_rad
				*/
				let inner_rad = 0.5 * (width - self.feathering);
				let outer_rad = 0.5 * (width + self.feathering);

				// Anti-alias the ends by extruding the outer edge and adding
				// two more triangles to each end:

				//   | aa |       | aa |
				//    _________________   ___
				//   | \    added    / |  feathering
				//   |   \ ___p___ /   |  ___
				//   |    |       |    |
				//   |    |  opa  |    |
				//   |    |  que  |    |
				//   |    |       |    |

				// (in the future it would be great with an option to add a circular end instead)

				// TODO(emilk): we should probably shrink before adding the line caps,
				// so that we don't add to the area of the line.
				// TODO(emilk): make line caps optional.
				let mut i0 = 0;
				let (bounds, _total_pts) = line.line_screen_points(t, |i1, p, n, color, is_last| {
					let i1 = i1 as u32;
					let is_first = i1 == 0;
					if is_first {
						let back_extrude = n.rot90() * self.feathering;
						out.colored_vertex(p + n * outer_rad + back_extrude, color_outer);
						out.colored_vertex(p + n * inner_rad, color);
						out.colored_vertex(p - n * inner_rad, color);
						out.colored_vertex(p - n * outer_rad + back_extrude, color_outer);

						out.add_triangle(idx, idx + 1, idx + 2);
						out.add_triangle(idx, idx + 2, idx + 3);
					} else if !is_last {
						out.colored_vertex(p + n * outer_rad, color_outer);
						out.colored_vertex(p + n * inner_rad, color);
						out.colored_vertex(p - n * inner_rad, color);
						out.colored_vertex(p - n * outer_rad, color_outer);

						out.add_triangle(idx + 4 * i0, idx + 4 * i0 + 1, idx + 4 * i1);
						out.add_triangle(idx + 4 * i0 + 1, idx + 4 * i1, idx + 4 * i1 + 1);

						out.add_triangle(idx + 4 * i0 + 1, idx + 4 * i0 + 2, idx + 4 * i1 + 1);
						out.add_triangle(idx + 4 * i0 + 2, idx + 4 * i1 + 1, idx + 4 * i1 + 2);

						out.add_triangle(idx + 4 * i0 + 2, idx + 4 * i0 + 3, idx + 4 * i1 + 2);
						out.add_triangle(idx + 4 * i0 + 3, idx + 4 * i1 + 2, idx + 4 * i1 + 3);
						i0 = i1;
					} else {
						let back_extrude = -n.rot90() * self.feathering;
						out.colored_vertex(p + n * outer_rad + back_extrude, color_outer);
						out.colored_vertex(p + n * inner_rad, color);
						out.colored_vertex(p - n * inner_rad, color);
						out.colored_vertex(p - n * outer_rad + back_extrude, color_outer);

						out.add_triangle(idx + 4 * i0, idx + 4 * i0 + 1, idx + 4 * i1);
						out.add_triangle(idx + 4 * i0 + 1, idx + 4 * i1, idx + 4 * i1 + 1);

						out.add_triangle(idx + 4 * i0 + 1, idx + 4 * i0 + 2, idx + 4 * i1 + 1);
						out.add_triangle(idx + 4 * i0 + 2, idx + 4 * i1 + 1, idx + 4 * i1 + 2);

						out.add_triangle(idx + 4 * i0 + 2, idx + 4 * i0 + 3, idx + 4 * i1 + 2);
						out.add_triangle(idx + 4 * i0 + 3, idx + 4 * i1 + 2, idx + 4 * i1 + 3);

						// The extension:
						out.add_triangle(idx + 4 * i1, idx + 4 * i1 + 1, idx + 4 * i1 + 2);
						out.add_triangle(idx + 4 * i1, idx + 4 * i1 + 2, idx + 4 * i1 + 3);
					}
				});

				bounds
			}
		}
	}
	fn tesselate_circle( pos: Pos2, radius: f32, color: Color32, out: &mut Mesh) {
		use precomputed_vertices::*;
		if radius <= 0.0 {
			return;
		}
		let precomputed: &[Vec2] = if radius <= 2.0 {
			CIRCLE_8.as_slice()
		} else if radius <= 5.0 {
			CIRCLE_16.as_slice()
		} else if radius <= 18.0 {
			CIRCLE_32.as_slice()
		} else if radius < 50.0 {
			CIRCLE_64.as_slice()
		} else {
			CIRCLE_128.as_slice()
		};
		let idx = out.vertices.len() as u32;
		for &n in precomputed.iter() {
			out.colored_vertex(pos + n * radius, color);
		}
		for i in 1..precomputed.len() as u32 - 1 {
			out.add_triangle(idx, idx + i, idx + i + 1);
		}
	}
}

enum LinePoints<'a> {
	Untransformed(&'a [(PlotPoint, Color32)]),
	Transformed(&'a [(Pos2, Color32)]),
}
impl<'a> LinePoints<'a> {
	fn len(&self) -> usize {
		match self {
			LinePoints::Untransformed(line) => line.len(),
			LinePoints::Transformed(line) => line.len(),
		}
	}

	fn line_screen_points(
		&self, t: &PlotTransform, mut cb: impl FnMut(usize, Pos2, Vec2, Color32, bool),
	) -> (Option<PlotBounds>, usize) {
		let n = self.len();
		let mut bounds = PlotBounds::NOTHING;
		match self {
			LinePoints::Untransformed(line) => {
				let mut n0 = (t.position_from_point(&line[1].0) - t.position_from_point(&line[0].0))
					.normalized()
					.rot90();
				bounds.extend_with(&line[0].0);
				cb(0, t.position_from_point(&line[0].0), n0, line[0].1, false);
				let mut total_pts = 1;
				for i in 1..n - 1 {
					bounds.extend_with(&line[i].0);
					let mut n1 = (t.position_from_point(&line[i + 1].0) - t.position_from_point(&line[i].0))
						.normalized()
						.rot90();

					// Handle duplicated points (but not triplicated…):
					if n0 == Vec2::ZERO {
						n0 = n1;
					} else if n1 == Vec2::ZERO {
						n1 = n0;
					}

					let normal = (n0 + n1) / 2.0;
					let length_sq = normal.length_sq();
					let right_angle_length_sq = 0.5;
					let sharper_than_a_right_angle = length_sq < right_angle_length_sq;
					if sharper_than_a_right_angle {
						// cut off the sharp corner
						let center_normal = normal.normalized();
						let n0c = (n0 + center_normal) / 2.0;
						let n1c = (n1 + center_normal) / 2.0;
						cb(
							total_pts,
							t.position_from_point(&line[i].0),
							n0c / n0c.length_sq(),
							line[i].1,
							false,
						);
						total_pts += 1;
						cb(
							total_pts,
							t.position_from_point(&line[i].0),
							n1c / n1c.length_sq(),
							line[i].1,
							false,
						);
						total_pts += 1;
					} else {
						// miter join
						cb(total_pts, t.position_from_point(&line[i].0), normal / length_sq, line[i].1, false);
						total_pts += 1;
					}

					n0 = n1;
				}

				bounds.extend_with(&line[n - 1].0);
				cb(
					total_pts,
					t.position_from_point(&line[n - 1].0),
					(t.position_from_point(&line[n - 1].0) - t.position_from_point(&line[n - 2].0))
						.normalized()
						.rot90(),
					line[n - 1].1,
					true,
				);
				total_pts += 1;
				(Some(bounds), total_pts)
			},
			LinePoints::Transformed(line) => {
				let mut n0 = (line[1].0 - line[0].0).normalized().rot90();
				cb(0, line[0].0, n0, line[0].1, false);
				let mut total_pts = 1;
				for i in 1..n - 1 {
					let mut n1 = (line[i + 1].0 - line[i].0).normalized().rot90();

					// Handle duplicated points (but not triplicated…):
					if n0 == Vec2::ZERO {
						n0 = n1;
					} else if n1 == Vec2::ZERO {
						n1 = n0;
					}

					let normal = (n0 + n1) / 2.0;
					let length_sq = normal.length_sq();
					let right_angle_length_sq = 0.5;
					let sharper_than_a_right_angle = length_sq < right_angle_length_sq;
					if sharper_than_a_right_angle {
						// cut off the sharp corner
						let center_normal = normal.normalized();
						let n0c = (n0 + center_normal) / 2.0;
						let n1c = (n1 + center_normal) / 2.0;
						cb(total_pts, line[i].0, n0c / n0c.length_sq(), line[i].1, false);
						total_pts += 1;
						cb(total_pts, line[i].0, n1c / n1c.length_sq(), line[i].1, false);
						total_pts += 1;
					} else {
						// miter join
						cb(total_pts, line[i].0, normal / length_sq, line[i].1, false);
						total_pts += 1;
					}

					n0 = n1;
				}
				cb(
					total_pts,
					line[n - 1].0,
					(line[n - 1].0 - line[n - 2].0).normalized().rot90(),
					line[n - 1].1,
					true,
				);
				total_pts += 1;
				(None, total_pts)
			},
		}
	}
}
#[expect(clippy::approx_constant)]
mod precomputed_vertices {
	// fn main() {
	//     let n = 64;
	//     println!("pub const CIRCLE_{}: [Vec2; {}] = [", n, n+1);
	//     for i in 0..=n {
	//         let a = std::f64::consts::TAU * i as f64 / n as f64;
	//         println!("    vec2({:.06}, {:.06}),", a.cos(), a.sin());
	//     }
	//     println!("];")
	// }

	use eframe::emath::{Vec2, vec2};

	pub const CIRCLE_8: [Vec2; 9] = [
		vec2(1.000000, 0.000000),
		vec2(0.707107, 0.707107),
		vec2(0.000000, 1.000000),
		vec2(-0.707107, 0.707107),
		vec2(-1.000000, 0.000000),
		vec2(-0.707107, -0.707107),
		vec2(0.000000, -1.000000),
		vec2(0.707107, -0.707107),
		vec2(1.000000, 0.000000),
	];

	pub const CIRCLE_16: [Vec2; 17] = [
		vec2(1.000000, 0.000000),
		vec2(0.923880, 0.382683),
		vec2(0.707107, 0.707107),
		vec2(0.382683, 0.923880),
		vec2(0.000000, 1.000000),
		vec2(-0.382684, 0.923880),
		vec2(-0.707107, 0.707107),
		vec2(-0.923880, 0.382683),
		vec2(-1.000000, 0.000000),
		vec2(-0.923880, -0.382683),
		vec2(-0.707107, -0.707107),
		vec2(-0.382684, -0.923880),
		vec2(0.000000, -1.000000),
		vec2(0.382684, -0.923879),
		vec2(0.707107, -0.707107),
		vec2(0.923880, -0.382683),
		vec2(1.000000, 0.000000),
	];

	pub const CIRCLE_32: [Vec2; 33] = [
		vec2(1.000000, 0.000000),
		vec2(0.980785, 0.195090),
		vec2(0.923880, 0.382683),
		vec2(0.831470, 0.555570),
		vec2(0.707107, 0.707107),
		vec2(0.555570, 0.831470),
		vec2(0.382683, 0.923880),
		vec2(0.195090, 0.980785),
		vec2(0.000000, 1.000000),
		vec2(-0.195090, 0.980785),
		vec2(-0.382683, 0.923880),
		vec2(-0.555570, 0.831470),
		vec2(-0.707107, 0.707107),
		vec2(-0.831470, 0.555570),
		vec2(-0.923880, 0.382683),
		vec2(-0.980785, 0.195090),
		vec2(-1.000000, 0.000000),
		vec2(-0.980785, -0.195090),
		vec2(-0.923880, -0.382683),
		vec2(-0.831470, -0.555570),
		vec2(-0.707107, -0.707107),
		vec2(-0.555570, -0.831470),
		vec2(-0.382683, -0.923880),
		vec2(-0.195090, -0.980785),
		vec2(-0.000000, -1.000000),
		vec2(0.195090, -0.980785),
		vec2(0.382683, -0.923880),
		vec2(0.555570, -0.831470),
		vec2(0.707107, -0.707107),
		vec2(0.831470, -0.555570),
		vec2(0.923880, -0.382683),
		vec2(0.980785, -0.195090),
		vec2(1.000000, -0.000000),
	];

	pub const CIRCLE_64: [Vec2; 65] = [
		vec2(1.000000, 0.000000),
		vec2(0.995185, 0.098017),
		vec2(0.980785, 0.195090),
		vec2(0.956940, 0.290285),
		vec2(0.923880, 0.382683),
		vec2(0.881921, 0.471397),
		vec2(0.831470, 0.555570),
		vec2(0.773010, 0.634393),
		vec2(0.707107, 0.707107),
		vec2(0.634393, 0.773010),
		vec2(0.555570, 0.831470),
		vec2(0.471397, 0.881921),
		vec2(0.382683, 0.923880),
		vec2(0.290285, 0.956940),
		vec2(0.195090, 0.980785),
		vec2(0.098017, 0.995185),
		vec2(0.000000, 1.000000),
		vec2(-0.098017, 0.995185),
		vec2(-0.195090, 0.980785),
		vec2(-0.290285, 0.956940),
		vec2(-0.382683, 0.923880),
		vec2(-0.471397, 0.881921),
		vec2(-0.555570, 0.831470),
		vec2(-0.634393, 0.773010),
		vec2(-0.707107, 0.707107),
		vec2(-0.773010, 0.634393),
		vec2(-0.831470, 0.555570),
		vec2(-0.881921, 0.471397),
		vec2(-0.923880, 0.382683),
		vec2(-0.956940, 0.290285),
		vec2(-0.980785, 0.195090),
		vec2(-0.995185, 0.098017),
		vec2(-1.000000, 0.000000),
		vec2(-0.995185, -0.098017),
		vec2(-0.980785, -0.195090),
		vec2(-0.956940, -0.290285),
		vec2(-0.923880, -0.382683),
		vec2(-0.881921, -0.471397),
		vec2(-0.831470, -0.555570),
		vec2(-0.773010, -0.634393),
		vec2(-0.707107, -0.707107),
		vec2(-0.634393, -0.773010),
		vec2(-0.555570, -0.831470),
		vec2(-0.471397, -0.881921),
		vec2(-0.382683, -0.923880),
		vec2(-0.290285, -0.956940),
		vec2(-0.195090, -0.980785),
		vec2(-0.098017, -0.995185),
		vec2(-0.000000, -1.000000),
		vec2(0.098017, -0.995185),
		vec2(0.195090, -0.980785),
		vec2(0.290285, -0.956940),
		vec2(0.382683, -0.923880),
		vec2(0.471397, -0.881921),
		vec2(0.555570, -0.831470),
		vec2(0.634393, -0.773010),
		vec2(0.707107, -0.707107),
		vec2(0.773010, -0.634393),
		vec2(0.831470, -0.555570),
		vec2(0.881921, -0.471397),
		vec2(0.923880, -0.382683),
		vec2(0.956940, -0.290285),
		vec2(0.980785, -0.195090),
		vec2(0.995185, -0.098017),
		vec2(1.000000, -0.000000),
	];

	pub const CIRCLE_128: [Vec2; 129] = [
		vec2(1.000000, 0.000000),
		vec2(0.998795, 0.049068),
		vec2(0.995185, 0.098017),
		vec2(0.989177, 0.146730),
		vec2(0.980785, 0.195090),
		vec2(0.970031, 0.242980),
		vec2(0.956940, 0.290285),
		vec2(0.941544, 0.336890),
		vec2(0.923880, 0.382683),
		vec2(0.903989, 0.427555),
		vec2(0.881921, 0.471397),
		vec2(0.857729, 0.514103),
		vec2(0.831470, 0.555570),
		vec2(0.803208, 0.595699),
		vec2(0.773010, 0.634393),
		vec2(0.740951, 0.671559),
		vec2(0.707107, 0.707107),
		vec2(0.671559, 0.740951),
		vec2(0.634393, 0.773010),
		vec2(0.595699, 0.803208),
		vec2(0.555570, 0.831470),
		vec2(0.514103, 0.857729),
		vec2(0.471397, 0.881921),
		vec2(0.427555, 0.903989),
		vec2(0.382683, 0.923880),
		vec2(0.336890, 0.941544),
		vec2(0.290285, 0.956940),
		vec2(0.242980, 0.970031),
		vec2(0.195090, 0.980785),
		vec2(0.146730, 0.989177),
		vec2(0.098017, 0.995185),
		vec2(0.049068, 0.998795),
		vec2(0.000000, 1.000000),
		vec2(-0.049068, 0.998795),
		vec2(-0.098017, 0.995185),
		vec2(-0.146730, 0.989177),
		vec2(-0.195090, 0.980785),
		vec2(-0.242980, 0.970031),
		vec2(-0.290285, 0.956940),
		vec2(-0.336890, 0.941544),
		vec2(-0.382683, 0.923880),
		vec2(-0.427555, 0.903989),
		vec2(-0.471397, 0.881921),
		vec2(-0.514103, 0.857729),
		vec2(-0.555570, 0.831470),
		vec2(-0.595699, 0.803208),
		vec2(-0.634393, 0.773010),
		vec2(-0.671559, 0.740951),
		vec2(-0.707107, 0.707107),
		vec2(-0.740951, 0.671559),
		vec2(-0.773010, 0.634393),
		vec2(-0.803208, 0.595699),
		vec2(-0.831470, 0.555570),
		vec2(-0.857729, 0.514103),
		vec2(-0.881921, 0.471397),
		vec2(-0.903989, 0.427555),
		vec2(-0.923880, 0.382683),
		vec2(-0.941544, 0.336890),
		vec2(-0.956940, 0.290285),
		vec2(-0.970031, 0.242980),
		vec2(-0.980785, 0.195090),
		vec2(-0.989177, 0.146730),
		vec2(-0.995185, 0.098017),
		vec2(-0.998795, 0.049068),
		vec2(-1.000000, 0.000000),
		vec2(-0.998795, -0.049068),
		vec2(-0.995185, -0.098017),
		vec2(-0.989177, -0.146730),
		vec2(-0.980785, -0.195090),
		vec2(-0.970031, -0.242980),
		vec2(-0.956940, -0.290285),
		vec2(-0.941544, -0.336890),
		vec2(-0.923880, -0.382683),
		vec2(-0.903989, -0.427555),
		vec2(-0.881921, -0.471397),
		vec2(-0.857729, -0.514103),
		vec2(-0.831470, -0.555570),
		vec2(-0.803208, -0.595699),
		vec2(-0.773010, -0.634393),
		vec2(-0.740951, -0.671559),
		vec2(-0.707107, -0.707107),
		vec2(-0.671559, -0.740951),
		vec2(-0.634393, -0.773010),
		vec2(-0.595699, -0.803208),
		vec2(-0.555570, -0.831470),
		vec2(-0.514103, -0.857729),
		vec2(-0.471397, -0.881921),
		vec2(-0.427555, -0.903989),
		vec2(-0.382683, -0.923880),
		vec2(-0.336890, -0.941544),
		vec2(-0.290285, -0.956940),
		vec2(-0.242980, -0.970031),
		vec2(-0.195090, -0.980785),
		vec2(-0.146730, -0.989177),
		vec2(-0.098017, -0.995185),
		vec2(-0.049068, -0.998795),
		vec2(-0.000000, -1.000000),
		vec2(0.049068, -0.998795),
		vec2(0.098017, -0.995185),
		vec2(0.146730, -0.989177),
		vec2(0.195090, -0.980785),
		vec2(0.242980, -0.970031),
		vec2(0.290285, -0.956940),
		vec2(0.336890, -0.941544),
		vec2(0.382683, -0.923880),
		vec2(0.427555, -0.903989),
		vec2(0.471397, -0.881921),
		vec2(0.514103, -0.857729),
		vec2(0.555570, -0.831470),
		vec2(0.595699, -0.803208),
		vec2(0.634393, -0.773010),
		vec2(0.671559, -0.740951),
		vec2(0.707107, -0.707107),
		vec2(0.740951, -0.671559),
		vec2(0.773010, -0.634393),
		vec2(0.803208, -0.595699),
		vec2(0.831470, -0.555570),
		vec2(0.857729, -0.514103),
		vec2(0.881921, -0.471397),
		vec2(0.903989, -0.427555),
		vec2(0.923880, -0.382683),
		vec2(0.941544, -0.336890),
		vec2(0.956940, -0.290285),
		vec2(0.970031, -0.242980),
		vec2(0.980785, -0.195090),
		vec2(0.989177, -0.146730),
		vec2(0.995185, -0.098017),
		vec2(0.998795, -0.049068),
		vec2(1.000000, -0.000000),
	];
}
