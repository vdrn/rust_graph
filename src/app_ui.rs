use std::sync::Mutex;

use eframe::egui::{
	self, Align, Button, CollapsingHeader, Id, RichText, ScrollArea, SidePanel, Slider, Stroke, TextEdit, TextStyle, Window
};
use eframe::epaint::Color32;
use egui_plot::{HLine, Legend, Plot, PlotImage, PlotPoint, VLine};
use evalexpr::EvalexprFloat;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::builtins::{init_builtins, show_builtin_information};
use crate::draw_buffer::{DrawMeshType, PointInteraction, PointInteractionType, process_draw_buffers};
use crate::entry::{self, Entry, EntryType, point_dragging, prepare_entries};
use crate::widgets::{duplicate_entry_btn, popup_label, remove_entry_btn};
use crate::{State, UiState, persistence, scope};

fn display_entry_errors(ui: &mut egui::Ui, ui_state: &UiState, entry_id: u64) {
	if let Some(parsing_error) = ui_state.parsing_errors.get(&entry_id) {
		ui.label(RichText::new(parsing_error).color(Color32::YELLOW));
	} else if let Some(preparation_error) = ui_state.prepare_errors.get(&entry_id) {
		ui.label(RichText::new(preparation_error).color(Color32::ORANGE));
	} else if let Some(optimization_error) = ui_state.optimization_errors.get(&entry_id) {
		ui.label(RichText::new(optimization_error).color(Color32::DARK_RED));
	} else if let Some(eval_error) = ui_state.eval_errors.get(&entry_id) {
		ui.label(RichText::new(eval_error).color(Color32::RED));
	}
}

pub fn side_panel<T: EvalexprFloat>(
	state: &mut State<T>, ui_state: &mut UiState, ctx: &egui::Context, frame: &mut eframe::Frame,
) {
	#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
	{
		ui_state.full_frame_scope.take();
		puffin::GlobalProfiler::lock().new_frame();
		ui_state.full_frame_scope = puffin::profile_scope_custom!("full_frame");
	}

	scope!("side_panel");
	SidePanel::left("left_panel").default_width(200.0).show(ctx, |ui| {
		ScrollArea::vertical().show(ui, |ui| {
			ui.add_space(10.0);
			ui.horizontal_top(|ui| {
				ui.heading("Rust Graph");
				if ui
					.button(if ui_state.conf.dark_mode { "🌙" } else { "☀" })
					.on_hover_text("Toggle Dark Mode")
					.clicked()
				{
					ui_state.conf.dark_mode = !ui_state.conf.dark_mode;
				}
			});

			ui.separator();

			let mut needs_recompilation = false;

			ui.horizontal_top(|ui| {
				if add_new_entry_btn(ui, "New entry", &mut ui_state.next_id, &mut state.entries, true) {
					needs_recompilation = true;
				}

				if ui.button("❌ Remove all").on_hover_text("Remove all entries").clicked() {
					state.entries.clear();
					state.clear_cache = true;
				}
			});

			ui.add_space(4.5);

			fn entry_frame(ui: &mut egui::Ui, is_dark_mode: bool, cb: impl FnOnce(&mut egui::Ui)) {
				let border_color = if is_dark_mode { Color32::from_gray(56) } else { Color32::from_gray(220) };
				let bg_color = if is_dark_mode { Color32::from_gray(32) } else { Color32::from_gray(240) };
				let shadow = egui::Shadow {
					offset: [0, 0],
					blur:   4,
					spread: 0,
					color:  if is_dark_mode { Color32::from_gray(65) } else { Color32::from_gray(195) },
				};

				egui::Frame::new()
					.inner_margin(4.0)
					.fill(bg_color)
					.shadow(shadow)
					.corner_radius(4.0)
					.stroke(Stroke::new(1.0, border_color))
					.show(ui, cb);
			}

			let mut remove = None;
			let mut duplicate = None;
			let mut animating = false;
			// println!("state.entries: {:?}", state.entries.iter().map(|e|e.id).collect::<Vec<_>>());
			egui_dnd::dnd(ui, "entries_dnd").show_vec(&mut state.entries, |ui, entry, handle, _state| {
				entry_frame(ui, ui_state.conf.dark_mode, |ui: &mut egui::Ui| {
					let symbol = entry.symbol();
					if let EntryType::Folder { entries } = &mut entry.ty {
						ui.vertical(|ui| {
							ui.horizontal(|ui| {
								handle.ui(ui, |ui| {
									ui.label("||").on_hover_text("Drag to reorder entries");
								});
								if ui
									.add(
										Button::new(RichText::new(symbol).strong().monospace())
											.corner_radius(10),
									)
									.on_hover_text(if entry.active { "Close folder" } else { "Open Folder" })
									.clicked()
								{
									entry.active = !entry.active;
								}

								ui.with_layout(egui::Layout::right_to_left(Align::LEFT), |ui| {
									if entries.is_empty() {
										if remove_entry_btn(ui, "Folder") {
											remove = Some(entry.id);
										}
									}
									if duplicate_entry_btn(ui, "Folder") {
										duplicate = Some(entry.id);
									}
									if add_new_entry_btn(ui, "", &mut ui_state.next_id, entries, false) {
										needs_recompilation = true;
									}
									ui.add(
										TextEdit::multiline(&mut entry.name)
											.hint_text("name")
											.desired_rows(1)
											.desired_width(ui.available_width()),
									)
									.changed();
								});
							});
							if !entry.active {
								return;
							}
							let mut remove_from_folder = None;
							let mut duplicate_in_folder = None;

							egui_dnd::dnd(ui, entry.id).show_vec(entries, |ui, entry, handle, _state| {
								entry_frame(ui, ui_state.conf.dark_mode, |ui: &mut egui::Ui| {
									ui.horizontal(|ui| {
										handle.ui(ui, |ui| {
											ui.label("    |").on_hover_text("Drag to reorder entries");
										});
										ui.horizontal(|ui| {
											let fe_result =
												entry::entry_ui(ui, &state.ctx, entry, state.clear_cache);
											if fe_result.remove {
												remove_from_folder = Some(entry.id);
											}
											if fe_result.duplicate {
												duplicate_in_folder = Some(entry.id);
											}
											animating |= fe_result.animating;
											needs_recompilation |= fe_result.needs_recompilation;
											if let Some(error) = fe_result.error {
												ui_state.parsing_errors.insert(entry.id, error);
											} else if fe_result.parsed {
												ui_state.parsing_errors.remove(&entry.id);
											}
										});
									});
									display_entry_errors(ui, ui_state, entry.id);
								});
							});
							if add_new_entry_btn(ui, "New Folder Entry", &mut ui_state.next_id, entries, false)
							{
								needs_recompilation = true;
							}
							if let Some(id) = remove_from_folder {
								if let Some(index) = entries.iter().position(|e| e.id == id) {
									entries.remove(index);
								}
							} else if let Some(id) = duplicate_in_folder {
								duplicate_entry(id, &mut ui_state.next_id, entries);
							}
						});
					} else {
						ui.horizontal(|ui| {
							handle.ui(ui, |ui| {
								ui.label("||").on_hover_text("Drag to reorder entries");
							});
							ui.horizontal(|ui| {
								let result = entry::entry_ui(ui, &state.ctx, entry, state.clear_cache);
								if result.remove {
									remove = Some(entry.id);
								}
								if result.duplicate {
									duplicate = Some(entry.id);
								}
								animating |= result.animating;
								needs_recompilation |= result.needs_recompilation;

								if let Some(error) = result.error {
									ui_state.parsing_errors.insert(entry.id, error);
								} else if result.parsed {
									ui_state.parsing_errors.remove(&entry.id);
								}
							});
						});
					}
					display_entry_errors(ui, ui_state, entry.id);
				});
			});
			if add_new_entry_btn(ui, "New Entry", &mut ui_state.next_id, &mut state.entries, true) {
				needs_recompilation = true;
			}

			if let Some(id) = remove {
				if let Some(index) = state.entries.iter().position(|e| e.id == id) {
					state.entries.remove(index);
				}
			} else if let Some(id) = duplicate {
				duplicate_entry(id, &mut ui_state.next_id, &mut state.entries);
			}
			ui.separator();

			#[cfg(not(target_arch = "wasm32"))]
			ui.hyperlink_to("View Online", {
				let mut base_url = "https://rust-graph.netlify.app/".to_string();
				base_url.push_str(ui_state.permalink_string.as_str());

				base_url
			});
			ui.separator();

			CollapsingHeader::new("Settings").default_open(true).show(ui, |ui| {
				ui.separator();
				ui.checkbox(&mut ui_state.conf.use_f32, "Use f32");

				ui.separator();
				ui.add(Slider::new(&mut ui_state.conf.resolution, 10..=2000).text("Point Resolution"));

				ui.separator();
				ui.add(Slider::new(&mut ui_state.conf.ui_scale, 1.0..=3.0).text("Ui Scale"));

				ui.separator();
				#[cfg(not(target_arch = "wasm32"))]
				if ui.button("Toggle Fullscreen").clicked()
					|| ui.input(|i| i.key_pressed(egui::Key::F11))
					|| ui.input(|i| i.modifiers.alt && i.key_pressed(egui::Key::Enter))
				{
					ui_state.conf.fullscreen = !ui_state.conf.fullscreen;
					ui.ctx().send_viewport_cmd(egui::ViewportCommand::Fullscreen(ui_state.conf.fullscreen));
				}

				if ui.button(format!("{} Help", if ui_state.showing_help { "📖" } else { "📚" })).clicked()
				{
					ui_state.showing_help = !ui_state.showing_help;
				}
				if ui_state.showing_help {
					Window::new("📖 Help").open(&mut ui_state.showing_help).show(ctx, |ui| {
						ScrollArea::vertical().show(ui, |ui| {
							show_builtin_information(ui);
						});
					});
				}
			});

			ui.separator();
			#[cfg(target_arch = "wasm32")]
			const PERSISTANCE_TYPE: &str = "Local Storage";
			#[cfg(not(target_arch = "wasm32"))]
			const PERSISTANCE_TYPE: &str = "Persistence";
			CollapsingHeader::new(PERSISTANCE_TYPE).default_open(true).show(ui, |ui| {
				persistence::persistence_ui(state, ui_state, ui, frame);
			});

			if let Some(error) = &ui_state.serialization_error {
				if ui.label(RichText::new(error).color(Color32::RED)).clicked() {
					ui_state.serialization_error = None;
				}
			}

			ui.separator();
			ui.hyperlink_to("Github", "https://github.com/vdrn/rust_graph");

			if needs_recompilation || state.clear_cache {
				scope!("prepare_entries");
				// state.points_cache.clear();
				match persistence::serialize_to_url(&state.entries, Some(&ui_state.plot_bounds)) {
					Ok(output) => {
						// let mut data_str = String::with_capacity(output.len() + 1);
						// let url_encoded = urlencoding::encode(str::from_utf8(&output).unwrap());
						let mut permalink_string = String::with_capacity(output.len() + 1);
						permalink_string.push('#');
						permalink_string.push_str(&output);
						ui_state.permalink_string = permalink_string;
						ui_state.scheduled_url_update = true;
					},
					Err(e) => {
						println!("Error: {e}");
					},
				}

				state.ctx.clear();
				init_builtins::<T>(&mut state.ctx);

				// state.context_stash.lock().unwrap().clear();

				prepare_entries(&mut state.entries, &mut state.ctx, &mut ui_state.prepare_errors);
			} else if animating {
				scope!("prepare_constants");
				entry::prepare_constants(&mut state.entries, &mut state.ctx, &mut ui_state.prepare_errors);
			}
			if needs_recompilation || state.clear_cache || animating {
				entry::optimize_entries(&mut state.entries, &mut state.ctx, &mut ui_state.optimization_errors);
			}
			state.clear_cache = false;

			if ui_state.scheduled_url_update {
				let time = ui.input(|i| i.time);

				if time - 1.0 > ui_state.last_url_update {
					ui_state.last_url_update = time;
					ui_state.scheduled_url_update = false;

					#[cfg(target_arch = "wasm32")]
					{
						use eframe::wasm_bindgen::prelude::*;
						let history = web_sys::window()
							.expect("Couldn't get window")
							.history()
							.expect("Couldn't get window.history");
						history
							.push_state_with_url(&JsValue::NULL, "", Some(&ui_state.permalink_string))
							.unwrap();
					}
				}
			}
		});
	});
	ui_state.eval_errors.clear();
}
pub fn graph_panel<T: EvalexprFloat>(
	state: &mut State<T>, ui_state: &mut UiState, ctx: &egui::Context, eframe: &eframe::Frame,
) {
	let first_x = ui_state.plot_bounds.min()[0];
	let last_x = ui_state.plot_bounds.max()[0];
	let first_y = ui_state.plot_bounds.min()[1];
	let last_y = ui_state.plot_bounds.max()[1];

	// let (first_x, last_x) = snap_range_to_grid(first_x, last_x, 10.0);
	let plot_width = last_x - first_x;
	let plot_height = last_y - first_y;

	let mut points_to_draw = ui_state.conf.resolution.max(1);
	let mut step_size = plot_width / points_to_draw as f64;
	while points_to_draw > 2 && first_x + step_size == first_x {
		points_to_draw /= 2;
		step_size = plot_width / points_to_draw as f64;
	}

	let mut points_to_draw_y = ui_state.conf.resolution.max(1);
	let mut step_size_y = plot_height / points_to_draw as f64;
	while points_to_draw_y > 2 && first_y + step_size_y == first_y {
		points_to_draw_y /= 2;
		step_size_y = plot_height / points_to_draw_y as f64;
	}
	let plot_params = entry::PlotParams {
		eps: if ui_state.conf.use_f32 { ui_state.f32_epsilon } else { ui_state.f64_epsilon },
		first_x,
		last_x,
		first_y,
		last_y,
		step_size,
		step_size_y,
		resolution: ui_state.conf.resolution,
		prev_plot_transform: ui_state.prev_plot_transform,
	};

	let main_context = &state.ctx;
	scope!("graph");

	let eval_errors = Mutex::new(&mut ui_state.eval_errors);

	state.entries.par_iter_mut().enumerate().for_each(|(i, entry)| {
		scope!("entry_draw", entry.name.clone());

		let id = Id::new(entry.id);
		if let Err(errors) = entry::entry_create_plot_elements(
			entry,
			id,
			i as u32 * 1000,
			ui_state.selected_plot_line.map(|(id, _)| id),
			main_context,
			&plot_params,
			&ui_state.draw_buffers,
			&state.thread_local_context,
		) {
			for (id, error) in errors {
				eval_errors.lock().unwrap().insert(id, error);
			}
		}
	});

	let mouse_pos_in_graph =
		if let (Some(pos), Some(trans)) = (ui_state.plot_mouese_pos, ui_state.prev_plot_transform) {
			let p = trans.value_from_position(pos);
			Some((p.x, p.y))
		} else {
			None
		};

	let mut p_draw_buffer = process_draw_buffers(
		ui_state.draw_buffers.as_mut(),
		ui_state.selected_plot_line,
		mouse_pos_in_graph,
		&plot_params,
	);

	egui::CentralPanel::default().show(ctx, |ui| {
		let plot_id = ui.make_persistent_id("Plot");
		let mut plot = Plot::new(plot_id)
			.id(plot_id)
			.legend(Legend::default().text_style(TextStyle::Body))
			.allow_drag(ui_state.dragging_point_i.is_none() && p_draw_buffer.closest_point_to_mouse.is_none());

		let mut bounds = ui_state.plot_bounds;
		if ui_state.reset_graph {
			ui_state.reset_graph = false;
			if let Some(default_bounds) = state.default_bounds {
				bounds = default_bounds;
			} else {
				plot = plot.center_x_axis(true).center_y_axis(true).data_aspect(1.0);
			}
		}
		plot = plot
			.default_x_bounds(bounds.min()[0], bounds.max()[0])
			.default_y_bounds(bounds.min()[1], bounds.max()[1]);
		if ui_state.showing_custom_label || p_draw_buffer.closest_point_to_mouse.is_some() {
			// plot = plot.label_formatter(|_, _| String::new());
			plot = plot.show_x(false);
			plot = plot.show_y(false);

			ui_state.showing_custom_label = false;
		}

		// plot.center_y_axis
		let mut hovered_point: Option<(bool, PointInteraction)> = None;

		if let (Some(custom_renderer), Some(prev_plot_transform)) =
			(&mut ui_state.custom_renderer, ui_state.prev_plot_transform)
		{
			// TODO: move this to process_draw_buffers
			let render_state = eframe.wgpu_render_state().unwrap();
			let draw_frame = prev_plot_transform.frame();

			let size = draw_frame.size();
			for mesh in p_draw_buffer.draw_meshes.iter_mut() {
				if let DrawMeshType::FillMesh(fill_mesh) = &mut mesh.ty {
					if fill_mesh.vertices.len() > 2 {
						fill_mesh.texture_id = Some(custom_renderer.paint_curve_fill(
							render_state, &fill_mesh.vertices, fill_mesh.color, size.x, size.y,
						));
					}
				}
			}
		}
		if let Some(custom_renderer) = &mut ui_state.custom_renderer {
			custom_renderer.reset_textures();
		}

		let plot_res = plot.show(ui, |plot_ui| {
			scope!("graph_show");
			plot_ui.hline(HLine::new("", 0.0).color(Color32::WHITE));
			plot_ui.vline(VLine::new("", 0.0).color(Color32::WHITE));
			for mesh in p_draw_buffer.draw_meshes {
				match mesh.ty {
					DrawMeshType::EguiPlotMesh(mesh) => {
						plot_ui.add(mesh);
					},
					DrawMeshType::FillMesh(fill_mesh) => {
						if let Some(texture_id) = fill_mesh.texture_id {
							let bounds = ui_state.plot_bounds;
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
				// plot_ui.mesh(mesh.mesh);
			}
			// if let Some(tex_id) = tex_id {
			// 	plot_ui.image(PlotImage::new("bla", tex_id, PlotPoint::new(0.0, 0.0), [2.0, 2.0]));
			// }
			for draw_poly_group in p_draw_buffer.draw_polygons {
				for poly in draw_poly_group.polygons {
					plot_ui.polygon(poly);
				}
			}
			for draw_line in p_draw_buffer.draw_lines {
				plot_ui.line(draw_line.line);
			}
			for draw_point in p_draw_buffer.draw_points {
				if let Some(mouse_pos) = ui_state.plot_mouese_pos {
					let transform = ui_state.prev_plot_transform.as_ref().unwrap();
					let sel = &draw_point.interaction;
					let is_draggable = matches!(sel.ty, PointInteractionType::Draggable { .. });
					let sel_p = transform.position_from_point(&PlotPoint::new(sel.x, sel.y));
					let dist_sq = (sel_p.x - mouse_pos.x).powf(2.0) + (sel_p.y - mouse_pos.y).powf(2.0);

					if dist_sq < sel.radius * sel.radius {
						if let Some(current_hovered) = &hovered_point {
							if !current_hovered.0 {
								hovered_point = Some((is_draggable, sel.clone()));
							}
						} else {
							hovered_point = Some((is_draggable, sel.clone()));
						}
					}
				}
				plot_ui.points(draw_point.points);
			}
			for draw_text in p_draw_buffer.draw_texts {
				plot_ui.text(draw_text.text);
			}
		});

		if plot_res.response.double_clicked() {
			ui_state.reset_graph = true;
		}

		if ui_state.plot_bounds != *plot_res.transform.bounds() {
			ui.ctx().request_repaint();
			ui_state.plot_bounds = *plot_res.transform.bounds();
		}
		ui_state.plot_mouese_pos = plot_res.response.hover_pos();
		ui_state.prev_plot_transform = Some(plot_res.transform);

		if let Some(drag_result) = point_dragging(
			&mut state.entries,
			&mut state.ctx,
			&plot_res,
			&mut ui_state.dragging_point_i,
			hovered_point.as_ref(),
			plot_params.eps,
		) {
			state.clear_cache = true;
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

		if let Some((closest_point, _dist_sq)) = p_draw_buffer.closest_point_to_mouse {
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
			&& let Some((_, hovered_point)) = &hovered_point
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

		if let Some(hovered_id) = plot_res.hovered_plot_item {
			if plot_res.response.clicked()
				|| plot_res.response.drag_started()
				|| (ui_state.selected_plot_line.is_none() && plot_res.response.is_pointer_button_down_on())
			{
				if state.entries.iter().any(|e| match &e.ty {
					EntryType::Function { .. } => Id::new(e.id) == hovered_id,
					EntryType::Folder { entries } => entries.iter().any(|e| Id::new(e.id) == hovered_id),
					_ => false,
				}) {
					ui_state.selected_plot_line = Some((hovered_id, true));
				}
			}
		} else {
			if plot_res.response.clicked() || plot_res.response.drag_started() {
				if let Some(selected_plot_line) = &mut ui_state.selected_plot_line {
					if !selected_plot_line.1 {
						ui_state.selected_plot_line = None;
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

fn prepare_copied_entry<T: EvalexprFloat>(entry: &mut Entry<T>, next_id: &mut u64) {
	if !entry.name.is_empty() {
		entry.name = format!("{}_c", entry.name);
	}
	*next_id += 1;
	entry.id = *next_id;
	if let EntryType::Folder { entries } = &mut entry.ty {
		for e in entries {
			prepare_copied_entry(e, next_id);
		}
	}
}
fn duplicate_entry<T: EvalexprFloat>(entry_id: u64, next_id: &mut u64, entries: &mut Vec<Entry<T>>) {
	let Some(entry_i) = entries.iter().position(|e| e.id == entry_id) else {
		return;
	};
	let entry = &entries[entry_i];
	let mut new_entry = entry.clone();
	prepare_copied_entry(&mut new_entry, next_id);
	entries.insert(entry_i + 1, new_entry);
}
fn add_new_entry_btn<T: EvalexprFloat>(
	ui: &mut egui::Ui, text: &str, next_id: &mut u64, entries: &mut Vec<Entry<T>>, can_add_folder: bool,
) -> bool {
	let mut needs_recompilation = false;

	if entries.len() >= 1000 {
		return false;
	}
	ui.menu_button(format!("➕ {text}"), |ui| {
		let new_function = Entry::new_function(*next_id, "");
		if ui.button(new_function.symbol_with_name()).clicked() {
			entries.push(new_function);
			*next_id += 1;
			needs_recompilation = true;
		}
		let new_constant = Entry::new_constant(*next_id);
		if ui.button(new_constant.symbol_with_name()).clicked() {
			entries.push(new_constant);
			*next_id += 1;
			needs_recompilation = true;
		}
		let new_points = Entry::new_points(*next_id);
		if ui.button(new_points.symbol_with_name()).clicked() {
			entries.push(new_points);
			*next_id += 1;
			needs_recompilation = true;
		}
		let new_label = Entry::new_label(*next_id);
		if ui.button(new_label.symbol_with_name()).clicked() {
			entries.push(new_label);
			*next_id += 1;
		}
		if can_add_folder {
			let new_folder = Entry::new_folder(*next_id);
			if ui.button(new_folder.symbol_with_name()).clicked() {
				entries.push(new_folder);
				*next_id += 1;
			}
		}
	})
	.response
	.on_hover_text(if can_add_folder { "Add new Entry" } else { "Add new Folder Entry" });

	needs_recompilation
}
