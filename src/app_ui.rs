use alloc::sync::Arc;

use eframe::egui::containers::menu::{MenuButton, MenuConfig};
use eframe::egui::{
	self, Align, Area, Button, CollapsingHeader, DragValue, Grid, Id, RichText, ScrollArea, SidePanel, Slider, Stroke, TextEdit, TextStyle, Window
};
use eframe::epaint::Color32;
use egui_plot::{HLine, Legend, Plot, PlotBounds, PlotImage, PlotPoint, PlotTransform, PlotUi, VLine};
use evalexpr::EvalexprFloat;
use serde::{Deserialize, Serialize};

use crate::builtins::{init_builtins, show_builtin_information};
use crate::draw_buffer::{DrawMeshType, PointInteraction, PointInteractionType, process_draw_buffers};
use crate::entry::{self, Entry, EntryType, point_dragging, prepare_entries, schedule_create_plot_elements};
use crate::widgets::{duplicate_entry_btn, popup_label, remove_entry_btn};
use crate::{AppConfig, State, UiState, persistence, scope};

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
) -> bool {
	#[cfg(all(feature = "puffin", not(target_arch = "wasm32")))]
	{
		ui_state.full_frame_scope.take();
		puffin::GlobalProfiler::lock().new_frame();
		ui_state.full_frame_scope = puffin::profile_scope_custom!("full_frame");
	}

	scope!("side_panel");
	let changed = SidePanel::left("left_panel")
		.default_width(200.0)
		.show(ctx, |ui| {
			ScrollArea::vertical()
				.show(ui, |ui| {
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
						if add_new_entry_btn(ui, "New entry", &mut ui_state.next_id, &mut state.entries, true)
						{
							needs_recompilation = true;
						}

						if ui.button("❌ Remove all").on_hover_text("Remove all entries").clicked() {
							state.entries.clear();
							state.clear_cache = true;
						}
					});

					ui.add_space(4.5);

					fn entry_frame(ui: &mut egui::Ui, is_dark_mode: bool, cb: impl FnOnce(&mut egui::Ui)) {
						let border_color =
							if is_dark_mode { Color32::from_gray(56) } else { Color32::from_gray(220) };
						let bg_color =
							if is_dark_mode { Color32::from_gray(32) } else { Color32::from_gray(240) };
						let shadow = egui::Shadow {
							offset: [0, 0],
							blur:   4,
							spread: 0,
							color:  if is_dark_mode {
								Color32::from_gray(65)
							} else {
								Color32::from_gray(195)
							},
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
					let mut needs_redraw = false;
					// println!("state.entries: {:?}", state.entries.iter().map(|e|e.id).collect::<Vec<_>>());
					egui_dnd::dnd(ui, "entries_dnd").show_vec(
						&mut state.entries,
						|ui, entry, handle, _state| {
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
												.on_hover_text(if entry.active {
													"Close folder"
												} else {
													"Open Folder"
												})
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
												if add_new_entry_btn(
													ui, "", &mut ui_state.next_id, entries, false,
												) {
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

										egui_dnd::dnd(ui, entry.id).show_vec(
											entries,
											|ui, entry, handle, _state| {
												entry_frame(
													ui,
													ui_state.conf.dark_mode,
													|ui: &mut egui::Ui| {
														ui.horizontal(|ui| {
															handle.ui(ui, |ui| {
																ui.label("    |")
																	.on_hover_text("Drag to reorder entries");
															});
															ui.horizontal(|ui| {
																let fe_result = entry::entry_ui(
																	ui, &state.ctx, &state.processed_colors,
																	entry, state.clear_cache,
																);
																if fe_result.remove {
																	remove_from_folder = Some(entry.id);
																}
																if fe_result.duplicate {
																	duplicate_in_folder = Some(entry.id);
																}
																needs_redraw |= fe_result.needs_redraw;
																animating |= fe_result.animating;
																needs_recompilation |=
																	fe_result.needs_recompilation;
																if let Some(error) = fe_result.error {
																	ui_state
																		.parsing_errors
																		.insert(entry.id, error);
																} else if fe_result.parsed {
																	ui_state.parsing_errors.remove(&entry.id);
																}
															});
														});
														display_entry_errors(ui, ui_state, entry.id);
													},
												);
											},
										);
										if add_new_entry_btn(
											ui, "New Folder Entry", &mut ui_state.next_id, entries, false,
										) {
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
											let result = entry::entry_ui(
												ui, &state.ctx, &state.processed_colors, entry,
												state.clear_cache,
											);
											if result.remove {
												remove = Some(entry.id);
											}
											if result.duplicate {
												duplicate = Some(entry.id);
											}
											needs_redraw |= result.needs_redraw;
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
						},
					);
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
							ui.ctx().send_viewport_cmd(egui::ViewportCommand::Fullscreen(
								ui_state.conf.fullscreen,
							));
						}

						if ui
							.button(format!("{} Help", if ui_state.showing_help { "📖" } else { "📚" }))
							.clicked()
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
						match persistence::serialize_to_url(&state.entries, ui_state.graph_config.clone()) {
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
						state.ctx.clear();
						init_builtins::<T>(&mut state.ctx);
						entry::prepare_constants(
							&mut state.entries, &mut state.ctx, &mut ui_state.prepare_errors,
						);
					}
					let changed_any = needs_recompilation || state.clear_cache || animating;
					if changed_any {
						state.processed_colors.clear();
						entry::optimize_entries(
							&mut state.entries,
							&mut state.ctx,
							&mut state.processed_colors,
							&mut ui_state.optimization_errors,
						);
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
					changed_any || needs_redraw
				})
				.inner
		})
		.inner;
	// ui_state.eval_errors.clear();
	changed
}
pub fn graph_panel<T: EvalexprFloat>(
	state: &mut State<T>, ui_state: &mut UiState, ctx: &egui::Context, eframe: &eframe::Frame, changed: bool,
) {
	let plot_params = entry::PlotParams::new::<T>(ui_state);

	scope!("graph");

	let mouse_pos_in_graph =
		if let (Some(pos), Some(trans)) = (ui_state.plot_mouese_pos, ui_state.prev_plot_transform) {
			let p = trans.value_from_position(pos);
			Some((p.x, p.y))
		} else {
			None
		};

	egui::CentralPanel::default().show(ctx, |ui| {
		let mut force_create_elements = false;

		if let Some(plot_transform) = ui_state.prev_plot_transform {
			let parent_rect = plot_transform.frame();
			let screen_pos = parent_rect.min + egui::vec2(5.0, 5.0);
			let prev_invert_axes = ui_state.graph_config.invert_axes;
			Area::new(Id::new("Graph Config")).fixed_pos(screen_pos).show(ui.ctx(), |ui| {
				MenuButton::new("⚙")
					.config(MenuConfig::new().close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside))
					.ui(ui, |ui| {
						ui_state.graph_config.ui(ui, &mut ui_state.conf);
					});
			});
			// if prev_show_axes != ui_state.graph_config.show_axes {
			// 	ui_state.force_process_elements = true;
			// }
			if prev_invert_axes != ui_state.graph_config.invert_axes {
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
							if let Some(entry) = get_entry_mut_by_id(&mut state.entries, id) {
								entry.draw_buffer = draw_buffer;
							}
						},
						Err((id, error)) => {
							if let Some(entry) = get_entry_mut_by_id(&mut state.entries, id) {
								ui_state.eval_errors.insert(id, error);
								entry.draw_buffer.clear();
							}
						},
					}
				}
			}

			if has_outstanding {
				ui.ctx().request_repaint();
			}
		}

		// TODO: this is not enough for changedete4ction.
		if changed || received_new || ui_state.reset_graph || ui_state.force_process_elements {
			scope!("process_draw_buffers");
			ui_state.processed_shapes.process(
				ui,
				&mut state.entries,
				&plot_params,
				ui_state.selected_plot_line.map(|(id, _)| id),
			);
			ui.ctx().request_repaint();
		}
		ui_state.force_process_elements = false;

		let p_draw_buffer = {
			scope!("process_draw_buffers");
			process_draw_buffers(
				&state.entries, ui_state.selected_plot_line, mouse_pos_in_graph, &plot_params,
				&mut ui_state.draw_buffers,
			)
		};

		let plot_id = ui.make_persistent_id("Plot");
		let mut plot = Plot::new(plot_id).id(plot_id).legend(Legend::default().text_style(TextStyle::Body));

		if ui_state.reset_graph {
			force_create_elements = true;
			ui_state.reset_graph = false;
			ui_state.graph_config = state.saved_graph_config.clone();
		}

		let available_size = ui.available_size();
		let view_aspect = available_size.x as f64 / available_size.y as f64;
		let calcd_bounds = ui_state.graph_config.graph_plot_bounds.calc_plot_bounds(view_aspect);

		let can_drag =
			ui_state.dragging_point_i.is_none() && p_draw_buffer.closest_point_to_mouse_on_selected.is_none();
		let mut allow_drag = ui_state.graph_config.allow_scroll;
		allow_drag[0] &= can_drag;
		allow_drag[1] &= can_drag;

		plot = plot
			.show_axes(ui_state.graph_config.show_axes)
			.invert_x(ui_state.graph_config.invert_axes[0])
			.invert_y(ui_state.graph_config.invert_axes[1])
			.show_grid(ui_state.graph_config.show_grid)
			.allow_drag(allow_drag)
			.allow_zoom(ui_state.graph_config.allow_zoom)
			.allow_scroll(ui_state.graph_config.allow_scroll)
			.x_axis_label(ui_state.graph_config.x_axis_label.clone())
			.y_axis_label(ui_state.graph_config.y_axis_label.clone())
			.clamp_grid(ui_state.graph_config.clamp_grid)
			.grid_spacing(ui_state.graph_config.grid_spacing.0..=ui_state.graph_config.grid_spacing.1);
		// .show_x(ui_state.graph_config.show_mouse_coords[0])
		// .show_y(ui_state.graph_config.show_mouse_coords[1]);

		plot = plot
			.default_x_bounds(calcd_bounds.min()[0], calcd_bounds.max()[0])
			.default_y_bounds(calcd_bounds.min()[1], calcd_bounds.max()[1]);

		if ui_state.showing_custom_label || p_draw_buffer.closest_point_to_mouse_on_selected.is_some() {
			// plot = plot.label_formatter(|_, _| String::new());
			plot = plot.show_x(false);
			plot = plot.show_y(false);

			ui_state.showing_custom_label = false;
		}

		// plot.center_y_axis
		let mut hovered_point: Option<(bool, PointInteraction)> = None;

		if let (Some(custom_renderer), Some(prev_plot_transform)) =
			(&mut ui_state.fan_fill_renderer, ui_state.prev_plot_transform)
		{
			// TODO: move this to process_draw_buffers
			let render_state = eframe.wgpu_render_state().unwrap();
			let draw_frame = prev_plot_transform.frame();

			let size = draw_frame.size();
			for mesh in ui_state.processed_shapes.draw_meshes.iter_mut() {
				if let DrawMeshType::FillMesh(fill_mesh) = &mut mesh.ty {
					if fill_mesh.vertices.len() > 2 {
						fill_mesh.texture_id = Some(custom_renderer.paint_curve_fill(
							render_state, &fill_mesh.vertices, &fill_mesh.indices, fill_mesh.color,
							fill_mesh.fill_rule, size.x, size.y,
						));
					}
				}
			}
		}
		if let Some(custom_renderer) = &mut ui_state.fan_fill_renderer {
			custom_renderer.reset_textures();
		}

		let plot_res = plot.show(ui, |plot_ui| {
			scope!("graph_show");

			if ui_state.graph_config.show_grid[0] {
				plot_ui.hline(HLine::new("", 0.0).color(Color32::WHITE));
			}
			if ui_state.graph_config.show_grid[1] {
				plot_ui.vline(VLine::new("", 0.0).color(Color32::WHITE));
			}
			for mesh in ui_state.processed_shapes.draw_meshes.iter() {
				match &mesh.ty {
					DrawMeshType::EguiPlotMesh(mesh) => {
						// TODO: we can have processed mesh instead, and keep it as Arc
						plot_ui.add(mesh.clone());
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
			}
			for draw_poly_group in ui_state.processed_shapes.draw_polygons.iter() {
				for poly in draw_poly_group.clone().polygons {
					plot_ui.polygon(poly);
				}
			}
			// for draw_line in p_draw_buffer.draw_lines {
			// 	plot_ui.line(draw_line.line);
			// }
			for line in ui_state.processed_shapes.lines.iter() {
				plot_ui.add(line.clone());
			}
			for draw_point in p_draw_buffer
				.draw_points
				.into_iter()
				.chain(ui_state.processed_shapes.draw_points.iter().cloned())
			{
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
			for draw_text in ui_state.processed_shapes.draw_texts.iter() {
				plot_ui.add(draw_text.text.clone());
			}

			ui_state.debug_info.draw(plot_ui);
		});

		if plot_res.response.double_clicked() {
			ui_state.reset_graph = true;
		}

		if ui_state.graph_config.graph_plot_bounds.update(&plot_res.transform, view_aspect) {
			force_create_elements = true;
		}

		if ui_state.plot_bounds != *plot_res.transform.bounds() {
			ui.ctx().request_repaint();
			ui_state.plot_bounds = *plot_res.transform.bounds();
		}
		if ui_state.prev_plot_transform.map(|t| *t.frame()) != Some(*plot_res.transform.frame()) {
			ui_state.force_process_elements = true;
			ui.ctx().request_repaint();
		}

		ui_state.plot_mouese_pos = plot_res.response.hover_pos();
		ui_state.prev_plot_transform = Some(plot_res.transform);

		if force_create_elements || changed {
			// println!("Scheduling element creation because force {force_create_elements} changed {changed}");
			if !ui_state.debug_info.pause_redraw {
				ui_state.debug_info.plot_bounds = Some(*plot_res.transform.bounds());
				scope!("schedule_entry_create_plot_elements");
				ui_state.eval_errors.clear();
				let plot_params = entry::PlotParams::new::<T>(ui_state);
				let main_context = Arc::new(state.ctx.clone());
				schedule_create_plot_elements(
					&mut state.entries,
					&mut ui_state.multi_draw_buffer_scheduler,
					ui_state.selected_plot_line.map(|(id, _)| id),
					&main_context,
					&plot_params,
					&state.thread_local_context,
				);
			}

			// for (i, entry) in state.entries.iter_mut().enumerate() {
			// 	schedule_entry_create_plot_elements(
			// 		entry,
			// 		i as u32 * 1000,
			// 		ui_state.selected_plot_line.map(|(id, _)| id),
			// 		&main_context,
			// 		&plot_params,
			// 		&state.thread_local_context,
			// 	);
			// }
			ui_state.force_process_elements = true;
		} else {
			let request_repaint = ui_state.multi_draw_buffer_scheduler.schedule_deffered_if_idle();
			if request_repaint {
				ui.ctx().request_repaint();
			}
		}

		if let Some(drag_result) = point_dragging(
			&mut state.entries,
			&mut state.ctx,
			&plot_res,
			&mut ui_state.dragging_point_i,
			hovered_point.as_ref(),
			&plot_params,
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

		if let Some(hovered_id) = plot_res.hovered_plot_item.or(p_draw_buffer.hovered_id)
			&& hovered_point.is_none()
		{
			// if dragging_point  {
			// ui_state.selected_plot_line = None;
			// }
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

fn prepare_copied_entry<T: EvalexprFloat>(entry: &mut Entry<T>, next_id: &mut u64) {
	if !entry.name.is_empty() {
		entry.name = format!("{}_c", entry.name);
	}
	entry.id = *next_id;
	*next_id += 1;
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
		let new_color = Entry::new_color(*next_id);
		if ui.button(new_color.symbol_with_name()).clicked() {
			entries.push(new_color);
			*next_id += 1;
			needs_recompilation = true;
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

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct GraphPlotBounds {
	pub center:      [f64; 2],
	pub h_size:      f64,
	pub data_aspect: f64,
}
impl Default for GraphPlotBounds {
	fn default() -> Self { Self { center: [0.0, 0.0], h_size: 4.0, data_aspect: 1.0 } }
}
impl GraphPlotBounds {
	fn calc_plot_bounds(&self, view_aspect: f64) -> PlotBounds {
		let v_size = self.h_size / self.data_aspect / view_aspect;
		let min_bounds = [self.center[0] - self.h_size * 0.5, self.center[1] - v_size * 0.5];
		let max_bounds = [self.center[0] + self.h_size * 0.5, self.center[1] + v_size * 0.5];
		PlotBounds::from_min_max(min_bounds, max_bounds)
	}

	pub fn update(&mut self, plot_transform: &PlotTransform, view_aspect: f64) -> bool {
		let prev = self.clone();

		let graph_bounds = plot_transform.bounds();
		self.center[0] = graph_bounds.center().x;
		self.center[1] = graph_bounds.center().y;
		self.h_size = graph_bounds.width();
		// let frame = plot_transform.frame();
		// let view_aspect = frame.size().x as f64 / frame.size().y as f64;
		// println!(
		// 	"view aspect {view_aspect} cur_data_aspect {} graph_aspect {}",
		// 	self.data_aspect,
		// 	graph_bounds.width() / graph_bounds.height()
		// );
		self.data_aspect = (graph_bounds.width() / graph_bounds.height()) / view_aspect;

		self != &prev
	}
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GraphConfig {
	pub graph_plot_bounds: GraphPlotBounds,
	pub allow_zoom:        [bool; 2],
	pub allow_scroll:      [bool; 2],
	// pub show_mouse_coords: [bool; 2],
	pub show_axes:         [bool; 2],
	pub invert_axes:       [bool; 2],
	pub show_grid:         [bool; 2],
	pub x_axis_label:      String,
	pub y_axis_label:      String,
	pub clamp_grid:        bool,
	pub grid_spacing:      (f32, f32),
}
impl Default for GraphConfig {
	fn default() -> Self {
		Self {
			graph_plot_bounds: GraphPlotBounds::default(),
			allow_zoom:        [true, true],
			allow_scroll:      [true, true],
			show_axes:         [true, true],
			show_grid:         [true, true],
			x_axis_label:      String::new(),
			y_axis_label:      String::new(),
			clamp_grid:        false, // show_mouse_coords: [true, true],
			grid_spacing:      (8.0, 300.0),
			invert_axes:       [false, false],
		}
	}
}
impl GraphConfig {
	fn ui(&mut self, ui: &mut egui::Ui, config: &mut AppConfig) {
		Grid::new("graph_config").num_columns(2).striped(true).show(ui, |ui| {
			ui.label("Allow Zoom");
			ui.horizontal(|ui| {
				ui.checkbox(&mut self.allow_zoom[0], "X");
				ui.checkbox(&mut self.allow_zoom[1], "Y");
			});
			ui.end_row();

			ui.label("Allow Scroll");
			ui.horizontal(|ui| {
				ui.checkbox(&mut self.allow_scroll[0], "X");
				ui.checkbox(&mut self.allow_scroll[1], "Y");
			});
			ui.end_row();

			ui.label("Show Axes");
			ui.horizontal(|ui| {
				ui.checkbox(&mut self.show_axes[0], "X");
				ui.checkbox(&mut self.show_axes[1], "Y");
			});
			ui.end_row();

			ui.label("Invert Axes");
			ui.horizontal(|ui| {
				ui.checkbox(&mut self.invert_axes[0], "X");
				ui.checkbox(&mut self.invert_axes[1], "Y");
			});

			ui.end_row();
			ui.label("X Axis Label");
			ui.text_edit_singleline(&mut self.x_axis_label);
			ui.end_row();

			ui.label("Y Axis Label");
			ui.text_edit_singleline(&mut self.y_axis_label);
			ui.end_row();

			ui.label("Show Grid");
			ui.horizontal(|ui| {
				ui.checkbox(&mut self.show_grid[0], "X");
				ui.checkbox(&mut self.show_grid[1], "Y");
			});
			ui.end_row();

			ui.label("Clamp Grid");
			ui.checkbox(&mut self.clamp_grid, "Clamp Grid")
				.on_hover_text("Only show grid where we have values");
			ui.end_row();

			ui.label("Grid Spacing");
			ui.horizontal(|ui| {
				const TOOLTIP: &str = "Set when the grid starts showing.
When grid lines are closer than the given minimum, they will be hidden.
When they get further apart they will fade in, until the reaches the given maximum,
at which point they are fully opaque.";
				ui.add(
					DragValue::new(&mut self.grid_spacing.0).range(1.0..=self.grid_spacing.1).prefix("Min: "),
				)
				.on_hover_text(TOOLTIP);
				ui.add(
					DragValue::new(&mut self.grid_spacing.1)
						.range(self.grid_spacing.0..=1000.0)
						.prefix("Max: "),
				)
				.on_hover_text(TOOLTIP);
			});
			ui.end_row();

			// ui.label("Show Mouse Coordinatess");
			// ui.checkbox(&mut self.show_mouse_coords[0], "X");
			// ui.checkbox(&mut self.show_mouse_coords[1], "Y");
			// ui.end_row();
		});
		ui.separator();
		ui.horizontal(|ui| {
			if ui.button("Make Default").clicked() {
				let bounds = config.default_graph_config.graph_plot_bounds.clone();
				config.default_graph_config = self.clone();
				config.default_graph_config.graph_plot_bounds = bounds;
			}
			if ui.button("Load default").clicked() {
				let bounds = self.graph_plot_bounds.clone();
				*self = config.default_graph_config.clone();
				self.graph_plot_bounds = bounds;
			}
			if ui.button("Reset").clicked() {
				let bounds = self.graph_plot_bounds.clone();
				*self = Default::default();
				self.graph_plot_bounds = bounds;
			}
			if ui.button("Close").clicked() {
				ui.close();
			}
		});
	}
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
	plot_bounds:  Option<PlotBounds>,
	pause_redraw: bool,
	draw:         bool,
}
impl DebugInfo {
	pub fn new() -> Self { Self { plot_bounds: None, pause_redraw: false, draw: false } }
	fn draw(&self, plot_ui: &mut PlotUi) {
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

			// if x >= bounds.min()[0] && x <= bounds.max()[0] {
			// 	let y = bounds.min()[1];
			// 	plot_ui.text(
			// 		Text::new("", PlotPoint::new(x, y), format!("{x:.2}")).color(Color32::from_gray(100)),
			// 	);
			// }
			// if y >= bounds.min()[1] && y <= bounds.max()[1] {
			// let x = bounds.min()[0];
			// plot_ui.text(
			// Text::new("", PlotPoint::new(x, y), format!("{y:.2}")).color(Color32::from_gray(100)),
			// );
			// }
		}
	}
}
