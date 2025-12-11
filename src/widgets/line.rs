use eframe::egui::{self, Color32, Id, Mesh, Pos2, StrokeKind, Vec2};
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
    base_color: Color32,
		series: Vec<(PlotPoint, Color32)>,

	) -> Self {
		Self { sorting_index, id, name, allow_hover: allow_hower, width, style, series, color: base_color }
	}
	pub fn points(&self) -> &Vec<(PlotPoint, Color32)> { &self.series }
	pub fn tessalate(&self, transform: &PlotTransform, out: &mut Mesh) -> PlotBounds {
		let mut tesselator = Tessellator {};
		let mut bounds = PlotBounds::NOTHING;
		match self.series.len() {
			0 => {},
			1 => {
				let radius = self.width * 0.5;
				// todo circle
			},
			_ => {
				let new_bounds = tesselator.tesselate_line(transform, &self.series, self.width, out);
				bounds.merge(&new_bounds);

				//todo dotted and datshed
			},
		}

		// let mut values_tf: Vec<Pos2> = Vec::with_capacity(self.series.len());
		// for (v, _) in self.series.iter() {
		// 	values_tf.push(transform.position_from_point(v));
		// 	bounds.extend_with(v)
		// }
		// let values_tf: Vec<Pos2> =
		// 	self.series.iter().map(|(v, _)| transform.position_from_point(v)).collect();
		// let stroke_kind = StrokeKind::Middle;

		// let mut shapes = Vec::new();
		// style_line(values_tf,self.style,  &self.series,self.width, mesh);

		bounds
	}
}

pub enum Shape2 {
	LineSegment([(Pos2, Color32); 2]),
	Circle(Pos2, f32, Color32),
}

pub(super) fn style_line(
	line: Vec<Pos2>, style: LineStyle, points: &[(PlotPoint, Color32)], width: f32, out: &mut Mesh,
) {
	match line.len() {
		0 => {},
		1 => {
			let mut radius = width / 2.0;
			// shapes.push(Shape2::Circle(line[0], radius, points[0].1));
		},
		_ => {
			// match self {
			// 	Self::Solid => {
			// 		shapes.push(Shape::line(line, stroke));
			// 	},
			// 	Self::Dotted { spacing } => {
			// 		// Take the stroke width for the radius even though it's not "correct", otherwise
			// 		// the dots would become too small.
			// 		let mut radius = width;
			// 		shapes.extend(Shape::dotted_line(&line, path_stroke_color, *spacing, radius));
			// 	},
			// 	Self::Dashed { length } => {
			// 		let golden_ratio = (5.0_f32.sqrt() - 1.0) / 2.0; // 0.61803398875
			// 		shapes.extend(Shape::dashed_line(
			// 			&line,
			// 			Stroke::new(stroke.width, path_stroke_color),
			// 			*length,
			// 			length * golden_ratio,
			// 		));
			// 	},
			// }
		},
	}
}

struct Tessellator {
	// clip_rect: Rect,
}
impl Tessellator {
	fn tesselate_line(
		&mut self, t: &PlotTransform, line: &[(PlotPoint, Color32)], width: f32, out: &mut Mesh,
	) -> PlotBounds {
		let idx = out.vertices.len() as u32;
		let n = line.len();
		if n < 2 || width <= 0.0 {
			return PlotBounds::NOTHING;
		}
		// let last_idx = n - 1;
		let radius = width * 0.5;
		let mut total_pts: u32 = 2;
		let mut bounds = PlotBounds::NOTHING;

		let mut n0 =
			(t.position_from_point(&line[1].0) - t.position_from_point(&line[0].0)).normalized().rot90();
		bounds.extend_with(&line[0].0);
		add_line_pt_vertices(out, t.position_from_point(&line[0].0), n0, line[0].1, width);
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
				add_line_pt_vertices(
					out,
					t.position_from_point(&line[i].0),
					n0c / n0c.length_sq(),
					line[i].1,
					radius,
				);
				add_line_pt_vertices(
					out,
					t.position_from_point(&line[i].0),
					n1c / n1c.length_sq(),
					line[i].1,
					radius,
				);
				total_pts += 2;
				// self.add_point(points[i], n0c / n0c.length_sq());
				// self.add_point(points[i], n1c / n1c.length_sq());
			} else {
				// miter join
				add_line_pt_vertices(
					out,
					t.position_from_point(&line[i].0),
					normal / length_sq,
					line[i].1,
					radius,
				);
				total_pts += 1;
				// self.add_point(points[i], normal / length_sq);
			}

			n0 = n1;
		}

		bounds.extend_with(&line[n - 1].0);
		add_line_pt_vertices(
			out,
			t.position_from_point(&line[n - 1].0),
			(t.position_from_point(&line[n - 1].0) - t.position_from_point(&line[n - 2].0))
				.normalized()
				.rot90(),
			line[0].1,
			radius,
		);

		out.reserve_triangles(2 * total_pts as usize);
		for i in 0..total_pts - 1 {
			out.add_triangle(
				idx + (2 * i + 0) % (2 * total_pts),
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
	}
}
fn add_line_pt_vertices(out: &mut Mesh, pt: Pos2, normal: egui::Vec2, color: Color32, radius: f32) {
	out.colored_vertex(pt + normal * radius, color);
	out.colored_vertex(pt - normal * radius, color);
}
