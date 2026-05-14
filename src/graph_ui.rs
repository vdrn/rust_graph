use alloc::sync::Arc;

use eframe::egui::containers::menu::{MenuButton, MenuConfig};
use eframe::egui::{self, Area, Color32, Id, TextStyle};
use egui_plot::{HLine, Legend, Plot, PlotBounds, PlotImage, PlotItem, PlotPoint, PlotUi, VLine};
use serde::{Deserialize, Serialize};

use evalexpr::EvalexprFloat;

use crate::entry::{Entry, EntryType};
use crate::graph_ui::create_plot_elements::{point_radius_outer, schedule_create_plot_elements};
use crate::graph_ui::graph_config::GraphConfig;
use crate::utils::popup_label;
use crate::{State, UiState, persistence, scope};

pub mod create_plot_elements;
pub mod graph_config;
pub mod plot_elements;
pub mod plot_elements_scheduler;
mod point_dragging;

use plot_elements::{DrawMeshType, PointInteraction, PointInteractionType};

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct IdGenerator {
	next_id: u64,
}
impl IdGenerator {
	pub fn next(&mut self) -> u64 {
		self.next_id += 1;
		self.next_id
	}
	pub fn new(start_id: u64) -> Self { Self { next_id: start_id } }
}

pub struct GraphState<T: EvalexprFloat> {
	pub entries:              Vec<Entry<T>>,
	pub saved_graph_config:   GraphConfig,
	pub current_graph_config: GraphConfig,
	pub name:                 String,
	pub id_gen:               IdGenerator,
	pub prev_plot_transform:  Option<egui_plot::PlotTransform>,
}
impl<T: EvalexprFloat> GraphState<T> {
	pub fn new(default_graph_config: GraphConfig) -> Self {
		let mut id_generator = IdGenerator::default();
		let first_id = id_generator.next();
		Self {
			entries:              vec![Entry::new_function(first_id, "sin(x)")],
			saved_graph_config:   default_graph_config.clone(),
			current_graph_config: default_graph_config,
			name:                 String::new(),
			id_gen:               id_generator,
			prev_plot_transform:  None,
		}
	}
	fn prev_plot_bounds(&self) -> PlotBounds {
		self.prev_plot_transform
			.as_ref()
			.map(|t| *t.bounds())
			.unwrap_or_else(|| PlotBounds::from_min_max([-2.0, -2.0], [2.0, 2.0]))
	}

	pub fn convert_float_type<TOUT: EvalexprFloat>(&self) -> Result<GraphState<TOUT>, String> {
		let name = self.name.clone();

		// TODO: very inefficient converting to json and back
		let mut ser_output = Vec::with_capacity(1024);
		persistence::serialize_graph_state_to_json(&mut ser_output, self)?;
		let mut graph_state = persistence::deserialize_graph_state_from_json(&ser_output)?;
		graph_state.name = name; // ensure name is preserved
		Ok(graph_state)
	}
}

#[derive(Clone)]
pub struct PlotParams {
	pub eps:                 f64,
	pub first_x:             f64,
	pub last_x:              f64,
	pub first_y:             f64,
	pub last_y:              f64,
	pub step_size:           f64,
	pub step_size_y:         f64,
	pub resolution:          usize,
	pub prev_plot_transform: Option<egui_plot::PlotTransform>,
	pub invert_axes:         [bool; 2],
}
impl PlotParams {
	pub fn new<T: EvalexprFloat>(ui_state: &UiState, graph_state: &GraphState<T>) -> Self {
		let plot_bounds = graph_state.prev_plot_bounds();
		let first_x = plot_bounds.min()[0];
		let last_x = plot_bounds.max()[0];
		let first_y = plot_bounds.min()[1];
		let last_y = plot_bounds.max()[1];

		// let (first_x, last_x) = snap_range_to_grid(first_x, last_x, 10.0);
		let plot_width = last_x - first_x;
		let plot_height = last_y - first_y;

		let mut points_to_draw = ui_state.app_config.resolution.max(1);
		let mut step_size = plot_width / points_to_draw as f64;
		while points_to_draw > 2 && first_x + step_size == first_x {
			points_to_draw /= 2;
			step_size = plot_width / points_to_draw as f64;
		}

		let mut points_to_draw_y = ui_state.app_config.resolution.max(1);
		let mut step_size_y = plot_height / points_to_draw as f64;
		while points_to_draw_y > 2 && first_y + step_size_y == first_y {
			points_to_draw_y /= 2;
			step_size_y = plot_height / points_to_draw_y as f64;
		}
		Self {
			eps: T::EPSILON,
			first_x,
			last_x,
			first_y,
			last_y,
			step_size,
			step_size_y,
			resolution: ui_state.app_config.resolution,
			prev_plot_transform: graph_state.prev_plot_transform,
			invert_axes: graph_state.current_graph_config.invert_axes,
		}
	}
}

pub fn graph_ui<T: evalexpr::EvalexprFloat>(
	ctx: &egui::Context, state: &mut State<T>, ui_state: &mut UiState, eframe: &eframe::Frame, changed: bool,
) {
	egui::CentralPanel::default().show(ctx, |ui| {
		scope!("graph");

		let mut force_create_elements = false;

		if let Some(plot_transform) = state.graph_state.prev_plot_transform {
			let parent_rect = plot_transform.frame();
			let screen_pos = parent_rect.min + egui::vec2(5.0, 5.0);
			let prev_invert_axes = state.graph_state.current_graph_config.invert_axes;
			Area::new(Id::new("Graph Config")).fixed_pos(screen_pos).show(ui.ctx(), |ui| {
				MenuButton::new("⚙")
					.config(MenuConfig::new().close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside))
					.ui(ui, |ui| {
						state.graph_state.current_graph_config.ui(ui, &mut ui_state.app_config);
					});
			});
			if prev_invert_axes != state.graph_state.current_graph_config.invert_axes {
				ui_state.force_process_elements = true;
				force_create_elements = true;
			}
		}

		if ui.input(|i| i.key_pressed(egui::Key::F9)) {
			ui_state.debug_info.pause_redraw = !ui_state.debug_info.pause_redraw;
		}
		if ui.input(|i| i.key_pressed(egui::Key::F10)) {
			ui_state.debug_info.draw = !ui_state.debug_info.draw;
		}

		let mut received_new = false;
		{
			scope!("receive_draw_buffers");
			let (has_outstanding, maybe_received) = ui_state.multi_draw_buffer_scheduler.try_receive();
			if let Some(received) = maybe_received {
				received_new = true;
				for r in received {
					match r {
						Ok((id, draw_buffer)) => {
							if let Some(entry) = get_entry_mut_by_id(&mut state.graph_state.entries, id) {
								ui_state.eval_errors.remove(&id);
								entry.raw_plot_elements = draw_buffer;
							}
						},
						Err((id, error)) => {
							if let Some(entry) = get_entry_mut_by_id(&mut state.graph_state.entries, id) {
								ui_state.eval_errors.insert(id, error);
								entry.raw_plot_elements.clear();
							}
						},
					}
				}
			}

			if has_outstanding {
				ui.ctx().request_repaint();
			}
		}

		let mouse_pos_in_graph = if let (Some(pos), Some(trans)) =
			(ui_state.plot_mouese_pos, state.graph_state.prev_plot_transform)
		{
			let p = trans.value_from_position(pos);
			Some((p.x, p.y))
		} else {
			None
		};

		let plot_params = PlotParams::new::<T>(ui_state, &state.graph_state);

		// TODO: this is not enough for changedete4ction.
		if changed || received_new || ui_state.reset_graph || ui_state.force_process_elements {
			scope!("process_draw_buffers");
			ui_state.processed_shapes.collect_and_process_draw_buffers(
				ui,
				eframe.wgpu_render_state().unwrap(),
				&mut ui_state.fan_fill_renderer,
				&mut state.graph_state.entries,
				&plot_params,
				ui_state.selected_plot_line.map(|(id, _)| id),
			);
			ui.ctx().request_repaint();
		}
		ui_state.force_process_elements = false;

		let p_draw_buffer = {
			scope!("process_draw_buffers");
			// TODO: this can be quite expensive for complex graphs, and most of it is not needed to be done
			// every frame, but only on user interaction (selecting lines)
			plot_elements::process_special_points(
				&state.graph_state.entries, ui_state.selected_plot_line, mouse_pos_in_graph, &plot_params,
				&mut ui_state.draw_buffers,
			)
		};

		let plot_id = ui.make_persistent_id("Plot");
		let mut plot = Plot::new(plot_id).id(plot_id);
		if state.graph_state.current_graph_config.show_legend {
			plot = plot.legend(Legend::default().text_style(TextStyle::Body));
		}

		if ui_state.reset_graph {
			force_create_elements = true;
			state.graph_state.current_graph_config = state.graph_state.saved_graph_config.clone();
		}

		let available_size = ui.available_size();
		let view_aspect = available_size.x as f64 / available_size.y as f64;

		let graph_conf = &state.graph_state.current_graph_config;
		let calcd_bounds = graph_conf.graph_plot_bounds.calc_plot_bounds(view_aspect);

		let can_drag =
			ui_state.dragging_point_i.is_none() && p_draw_buffer.closest_point_to_mouse_on_selected.is_none();
		let mut allow_drag = graph_conf.allow_scroll;
		allow_drag[0] &= can_drag;
		allow_drag[1] &= can_drag;

		plot = plot
			.show_axes(graph_conf.show_axes)
			.invert_x(graph_conf.invert_axes[0])
			.invert_y(graph_conf.invert_axes[1])
			.show_grid(graph_conf.show_grid)
			.allow_drag(allow_drag)
			.allow_zoom(graph_conf.allow_zoom)
			.allow_scroll(graph_conf.allow_scroll)
			.x_axis_label(graph_conf.x_axis_label.clone())
			.y_axis_label(graph_conf.y_axis_label.clone())
			.clamp_grid(graph_conf.clamp_grid)
			.grid_spacing(graph_conf.grid_spacing.0..=graph_conf.grid_spacing.1);

		if ui_state.reset_graph {
			plot = plot.reset();
			ui_state.reset_graph = false;
		}
		plot = plot
			.default_x_bounds(calcd_bounds.min()[0], calcd_bounds.max()[0])
			.default_y_bounds(calcd_bounds.min()[1], calcd_bounds.max()[1]);

		if ui_state.showing_custom_label || p_draw_buffer.closest_point_to_mouse_on_selected.is_some() {
			plot = plot.show_x(false);
			plot = plot.show_y(false);

			ui_state.showing_custom_label = false;
		}

		let mut hovered_point: Option<(bool, PointInteraction, f32)> = None;

		let prev_plot_bounds = state.graph_state.prev_plot_bounds();

		let plot_res = plot.show(ui, |plot_ui| {
			scope!("graph_show");

			if state.graph_state.current_graph_config.show_grid[0] {
				plot_ui.hline(HLine::new("", 0.0).color(Color32::WHITE));
			}
			if state.graph_state.current_graph_config.show_grid[1] {
				plot_ui.vline(VLine::new("", 0.0).color(Color32::WHITE));
			}
			for mesh in ui_state.processed_shapes.draw_meshes.iter() {
				match &mesh.ty {
					DrawMeshType::EguiPlotMesh(mesh) => {
						plot_ui.add(mesh.clone());
					},
					DrawMeshType::FillMesh(fill_mesh) => {
						if let Some(texture_id) = fill_mesh.texture_id {
							let bounds = prev_plot_bounds;
							let center = bounds.center();
							let size_x = bounds.width();
							let size_y = bounds.height();

							plot_ui.image(PlotImage::new(
								"",
								texture_id,
								center,
								[size_x as f32, size_y as f32],
							));
						}
					},
				}
			}
			for draw_poly_group in ui_state.processed_shapes.draw_polygons.iter() {
				for poly in draw_poly_group.clone().polygons {
					plot_ui.polygon(poly);
				}
			}
			for line in ui_state.processed_shapes.lines.iter() {
				plot_ui.add(line.clone());
			}
			for draw_point in
				p_draw_buffer.draw_points.iter().chain(ui_state.processed_shapes.draw_points.iter())
			{
				if let Some(mouse_pos) = ui_state.plot_mouese_pos {
					let transform = state.graph_state.prev_plot_transform.as_ref().unwrap();
					let sel = &draw_point.interaction;
					let is_draggable = matches!(sel.ty, PointInteractionType::Draggable { .. });
					let sel_p = transform.position_from_point(&PlotPoint::new(sel.x, sel.y));
					let dist_sq = (sel_p.x - mouse_pos.x).powf(2.0) + (sel_p.y - mouse_pos.y).powi(2);

					let hover_radius = if is_draggable { point_radius_outer(false) } else { sel.radius };
					if dist_sq < hover_radius.powi(2) {
						if let Some(current_hovered) = &hovered_point {
							let replace_current = match (current_hovered.0, is_draggable) {
								(false, true) => true,
								(true, false) => false,
								(false, false) | (true, true) => current_hovered.2 > dist_sq,
							};
							if replace_current {
								hovered_point = Some((is_draggable, sel.clone(), dist_sq));
							}
						} else {
							hovered_point = Some((is_draggable, sel.clone(), dist_sq));
						}
					}
				}
			}
			let dragging_or_hovered_id =
				ui_state.dragging_point_i.as_ref().or(hovered_point.as_ref().map(|h| &h.1)).and_then(|dp| {
					if let PointInteractionType::Draggable { i, .. } = dp.ty {
						Some(i.0.with(i.1))
					} else {
						None
					}
				});

			for mut draw_point in p_draw_buffer
				.draw_points
				.into_iter()
				.chain(ui_state.processed_shapes.draw_points.iter().cloned())
			{
				if let Some(id) = dragging_or_hovered_id {
					if (&draw_point.points as &dyn PlotItem).id() == id {
						let is_selected = Some(id) == ui_state.selected_plot_line.map(|i| i.0);
						let radius = point_radius_outer(is_selected);
						draw_point.points = draw_point.points.radius(radius);
					}
				}
				plot_ui.points(draw_point.points);
			}
			for draw_text in ui_state.processed_shapes.draw_texts.iter() {
				plot_ui.add(draw_text.text.clone());
			}

			ui_state.debug_info.draw(plot_ui);
		});

		if plot_res.response.double_clicked() {
			ui_state.reset_graph = true;
		}

		if state.graph_state.current_graph_config.graph_plot_bounds.update(&plot_res.transform, view_aspect) {
			force_create_elements = true;
		}

		if prev_plot_bounds != *plot_res.transform.bounds() {
			ui.ctx().request_repaint();
		}
		if state.graph_state.prev_plot_transform.map(|t| *t.frame()) != Some(*plot_res.transform.frame()) {
			ui_state.force_process_elements = true;
			ui.ctx().request_repaint();
		}

		ui_state.plot_mouese_pos = plot_res.response.hover_pos();
		state.graph_state.prev_plot_transform = Some(plot_res.transform);

		if force_create_elements || changed {
			if !ui_state.debug_info.pause_redraw {
				ui_state.debug_info.plot_bounds = Some(*plot_res.transform.bounds());
				scope!("schedule_entry_create_plot_elements");

				ui_state.eval_errors.clear();
				let plot_params = PlotParams::new::<T>(ui_state, &state.graph_state);
				let main_context = Arc::new(state.ctx.clone());
				schedule_create_plot_elements(
					&mut state.graph_state.entries,
					&mut ui_state.multi_draw_buffer_scheduler,
					ui_state.selected_plot_line.map(|(id, _)| id),
					&main_context,
					&plot_params,
					&state.thread_local_context,
					&state.processed_colors,
				);
			}

			ui_state.force_process_elements = true;
		} else {
			let request_repaint = ui_state.multi_draw_buffer_scheduler.schedule_deffered_if_idle();
			if request_repaint {
				ui.ctx().request_repaint();
			}
		}

		if let Some(drag_result) = point_dragging::drag_point(
			&mut state.graph_state.entries,
			&mut state.ctx,
			&plot_res,
			&mut ui_state.dragging_point_i,
			hovered_point.as_ref(),
			&plot_params,
		) {
			ui_state.clear_cache = true;
			ui_state.showing_custom_label = true;
			let screen_x = plot_res.transform.position_from_point_x(drag_result.x);
			let screen_y = plot_res.transform.position_from_point_y(drag_result.y);
			popup_label(
				ui,
				Id::new("drag_point_popup"),
				format!(
					"Point {}\nx: {}\ny: {}",
					drag_result.name,
					drag_result.x.human_display(false),
					drag_result.y.human_display(false)
				),
				[screen_x, screen_y],
			);
		}

		if let Some((closest_point, _dist_sq)) = p_draw_buffer.closest_point_to_mouse_on_selected {
			let screen_x = plot_res.transform.position_from_point_x(closest_point.0);
			let screen_y = plot_res.transform.position_from_point_y(closest_point.1);
			ui_state.showing_custom_label = true;
			popup_label(
				ui,
				Id::new("point_on_fn"),
				format!(
					"x:{}\ny: {}",
					T::from_f64(closest_point.0).human_display(false),
					T::from_f64(closest_point.1).human_display(false)
				),
				[screen_x, screen_y],
			);
		}

		if !ui_state.showing_custom_label
			&& let Some((_, hovered_point, _)) = &hovered_point
		{
			let screen_x = plot_res.transform.position_from_point_x(hovered_point.x);
			let screen_y = plot_res.transform.position_from_point_y(hovered_point.y);
			ui_state.showing_custom_label = true;
			popup_label(
				ui,
				Id::new("point_popup"),
				format!(
					"{}\nx:{}\ny: {}",
					hovered_point.name(),
					T::from_f64(hovered_point.x).human_display(false),
					T::from_f64(hovered_point.y).human_display(false)
				),
				[screen_x, screen_y],
			);
		}

		if let Some(hovered_id) = plot_res.hovered_plot_item.or(p_draw_buffer.hovered_id)
			&& hovered_point.is_none()
		{
			if plot_res.response.clicked()
				|| plot_res.response.drag_started()
				|| (ui_state.selected_plot_line.is_none() && plot_res.response.is_pointer_button_down_on())
			{
				if state.graph_state.entries.iter().any(|e| match &e.ty {
					EntryType::Function { .. } => Id::new(e.id) == hovered_id,
					EntryType::Folder { entries } => entries.iter().any(|e| Id::new(e.id) == hovered_id),
					_ => false,
				}) {
					ui_state.selected_plot_line = Some((hovered_id, true));
					ui_state.force_process_elements = true;
				}
			}
		} else {
			if plot_res.response.clicked() || plot_res.response.drag_started() {
				if let Some(selected_plot_line) = &mut ui_state.selected_plot_line {
					if !selected_plot_line.1 {
						ui_state.selected_plot_line = None;
						ui_state.force_process_elements = true;
					}
				}
			}
		}

		if let Some(selected_plot_line) = &mut ui_state.selected_plot_line {
			if selected_plot_line.1 && !plot_res.response.is_pointer_button_down_on() {
				selected_plot_line.1 = false;
			}
		}
	});
}
fn get_entry_mut_by_id<T: EvalexprFloat>(entries: &mut [Entry<T>], id: u64) -> Option<&mut Entry<T>> {
	for entry in entries.iter_mut() {
		if entry.id == id {
			return Some(entry);
		} else if let EntryType::Folder { entries } = &mut entry.ty {
			for sub_entry in entries.iter_mut() {
				if sub_entry.id == id {
					return Some(sub_entry);
				}
			}
		}
	}
	None
}

pub struct DebugInfo {
	pub plot_bounds:  Option<PlotBounds>,
	pub pause_redraw: bool,
	pub draw:         bool,
}
impl DebugInfo {
	pub fn new() -> Self { Self { plot_bounds: None, pause_redraw: false, draw: false } }
	pub fn draw(&self, plot_ui: &mut PlotUi) {
		let _bounds = plot_ui.plot_bounds();
		if !self.draw {
			return;
		}
		let Some(last_bounds) = self.plot_bounds else {
			return;
		};
		let reso = 300;
		let width = last_bounds.width();
		let height = last_bounds.height();
		let cgrid_w = width / reso as f64;
		let cgrid_h = height / reso as f64;
		for i in 0..=reso {
			let x = last_bounds.min()[0] + cgrid_w * i as f64;
			let y = last_bounds.min()[1] + cgrid_h * i as f64;
			plot_ui.hline(HLine::new("", y).color(Color32::from_gray(150)));
			plot_ui.vline(VLine::new("", x).color(Color32::from_gray(150)));

			let x_mid = last_bounds.min()[0] + cgrid_w * i as f64 + cgrid_w * 0.5;
			let y_mid = last_bounds.min()[1] + cgrid_h * i as f64 + cgrid_h * 0.5;
			plot_ui.hline(HLine::new("", y_mid).width(0.6).color(Color32::from_gray(100)));
			plot_ui.vline(VLine::new("", x_mid).width(0.6).color(Color32::from_gray(100)));
		}
	}
}
