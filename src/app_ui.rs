use std::sync::Mutex;

use eframe::egui::{
	self, Align, Button, CollapsingHeader, Id, RichText, ScrollArea, SidePanel, Slider, TextEdit, TextStyle, Window
};
use eframe::epaint::Color32;
use egui_plot::{HLine, Legend, Plot, PlotPoint, VLine};
use evalexpr::EvalexprFloat;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::draw_buffer::{PointInteraction, PointInteractionType, process_draw_buffers};
use crate::entry::{self, Entry, EntryType, f64_to_float, point_dragging};
use crate::{
	BOOLEAN_OPERATORS, BUILTIN_CONSTS, BUILTIN_FUNCTIONS, OPERATORS, State, UiState, init_consts, init_functions, persistence, scope
};

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
				if ui.button(if ui_state.conf.dark_mode { "🌙" } else { "☀" }).clicked() {
					ui_state.conf.dark_mode = !ui_state.conf.dark_mode;
				}
			});

			ui.separator();

			let mut needs_recompilation = false;

			ui.horizontal_top(|ui| {
				if add_new_entry_btn(ui, &mut ui_state.next_id, &mut state.entries, true) {
					needs_recompilation = true;
				}

				if ui.button("❌ Clear all").clicked() {
					state.entries.clear();
					state.clear_cache = true;
				}
			});

			ui.add_space(4.5);

			let mut remove = None;
			let mut animating = false;
			// println!("state.entries: {:?}", state.entries.iter().map(|e|e.id).collect::<Vec<_>>());
			egui_dnd::dnd(ui, "entries_dnd").show_vec(&mut state.entries, |ui, entry, handle, _state| {
				if let EntryType::Folder { entries } = &mut entry.ty {
					ui.vertical(|ui| {
						ui.horizontal(|ui| {
							handle.ui(ui, |ui| {
								ui.label("||");
							});
							let folder_symbol = if entry.visible { "📂" } else { "📁" };
							if ui
								.add(
									Button::new(RichText::new(folder_symbol).strong().monospace())
										.corner_radius(10),
								)
								.clicked()
							{
								entry.visible = !entry.visible;
							}

							ui.with_layout(egui::Layout::right_to_left(Align::LEFT), |ui| {
								if entries.is_empty() {
									if ui.button("X").clicked() {
										remove = Some(entry.id);
									}
								}
								if add_new_entry_btn(ui, &mut ui_state.next_id, entries, false) {
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
						if entry.visible {
							let mut remove_from_folder = None;

							egui_dnd::dnd(ui, entry.id).show_vec(entries, |ui, entry, handle, _state| {
								ui.horizontal(|ui| {
									handle.ui(ui, |ui| {
										ui.label("    |");
									});
									ui.horizontal(|ui| {
										let fe_result = entry::entry_ui(ui, entry, state.clear_cache);
										if fe_result.remove {
											remove_from_folder = Some(entry.id);
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
								if let Some(parsing_error) = ui_state.parsing_errors.get(&entry.id) {
									ui.label(RichText::new(parsing_error).color(Color32::RED));
								} else if let Some(eval_error) = ui_state.eval_errors.get(&entry.id) {
									ui.label(RichText::new(eval_error).color(Color32::RED));
								}
							});
							if let Some(id) = remove_from_folder {
								if let Some(index) = entries.iter().position(|e| e.id == id) {
									entries.remove(index);
								}
							}
						}
					});
				} else {
					ui.horizontal(|ui| {
						handle.ui(ui, |ui| {
							ui.label("||");
						});
						ui.horizontal(|ui| {
							let result = entry::entry_ui(ui, entry, state.clear_cache);
							if result.remove {
								remove = Some(entry.id);
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
				if let Some(parsing_error) = ui_state.parsing_errors.get(&entry.id) {
					ui.label(RichText::new(parsing_error).color(Color32::RED));
				} else if let Some(eval_error) = ui_state.eval_errors.get(&entry.id) {
					ui.label(RichText::new(eval_error).color(Color32::RED));
				}
				ui.separator();
			});

			if let Some(id) = remove {
				if let Some(index) = state.entries.iter().position(|e| e.id == id) {
					state.entries.remove(index);
				}
			}

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
							ui.columns_const::<3, _>(|columns| {
								columns[0].heading("Operators");
								for &(name, value) in OPERATORS {
									columns[0].horizontal_wrapped(|ui| {
										ui.label(RichText::new(name).monospace().strong());
										ui.label(value);
									});
								}
								columns[1].heading("Boolean Operators");
								for &(name, value) in BOOLEAN_OPERATORS {
									columns[1].horizontal_wrapped(|ui| {
										ui.label(RichText::new(name).monospace().strong());
										ui.label(value);
									});
								}
								columns[2].heading("Builtin Constants");
								for &(name, value) in BUILTIN_CONSTS {
									columns[2].horizontal_wrapped(|ui| {
										ui.label(RichText::new(name).monospace().strong());
										ui.label(value);
									});
								}
							});

							ui.separator();
							ui.heading("Builtin Functions");
							for &(name, value) in BUILTIN_FUNCTIONS {
								if name.is_empty() {
									ui.separator();
								} else {
									ui.horizontal_wrapped(|ui| {
										ui.label(RichText::new(name).monospace().strong());
										ui.label(value);
									});
								}
							}
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
				scope!("clear_cache");
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
				init_functions::<T>(&mut state.ctx);
				init_consts::<T>(&mut state.ctx);

				// state.context_stash.lock().unwrap().clear();

				for entry in state.entries.iter_mut() {
					scope!("entry_recompile");
					if let Err((id, e)) = entry::prepare_entry(entry, &mut state.ctx) {
						ui_state.parsing_errors.insert(id, e);
					}
				}
			} else if animating {
				for entry in state.entries.iter_mut() {
					scope!("entry_process");
					match &mut entry.ty {
						EntryType::Constant { value, istr_name, .. } => {
							if !entry.name.is_empty() {
								state.ctx.set_value(*istr_name, evalexpr::Value::<T>::Float(*value)).unwrap();
							}
						},
						EntryType::Folder { entries } => {
							for entry in entries {
								if let EntryType::Constant { value, istr_name, .. } = &mut entry.ty {
									if !entry.name.is_empty() {
										state
											.ctx
											.set_value(*istr_name, evalexpr::Value::<T>::Float(*value))
											.unwrap();
									}
								}
							}
						},
						_ => {},
					}
				}
			}
			if needs_recompilation || state.clear_cache || animating {
				entry::optimize_entries(&mut state.entries, &mut state.ctx, &mut ui_state.parsing_errors);
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
pub fn graph_panel<T: EvalexprFloat>(state: &mut State<T>, ui_state: &mut UiState, ctx: &egui::Context) {
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

	let mouse_pos_in_graph = ui_state.plot_mouese_pos.map(|(pos, trans)| {
		let p = trans.value_from_position(pos);
		(p.x, p.y)
	});

	let p_draw_buffer = process_draw_buffers(
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
			plot = plot.data_aspect(1.0).center_x_axis(true).center_y_axis(true).reset();
			if let Some(default_bounds) = state.default_bounds {
				bounds = default_bounds;
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

		let plot_res = plot.show(ui, |plot_ui| {
			scope!("graph_show");
			plot_ui.hline(HLine::new("", 0.0).color(Color32::WHITE));
			plot_ui.vline(VLine::new("", 0.0).color(Color32::WHITE));
			for draw_poly_group in p_draw_buffer.draw_polygons {
				for poly in draw_poly_group.polygons {
					plot_ui.polygon(poly);
				}
			}
			for draw_line in p_draw_buffer.draw_lines {
				plot_ui.line(draw_line.line);
			}
			for draw_point in p_draw_buffer.draw_points {
				if let Some((mouse_pos, transform)) = ui_state.plot_mouese_pos {
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

		ui_state.plot_bounds = *plot_res.transform.bounds();
		ui_state.plot_mouese_pos = plot_res.response.hover_pos().map(|p| (p, plot_res.transform));

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
			show_popup_label(
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
			show_popup_label(
				ui,
				Id::new("point_on_fn"),
				format!(
					"x:{}\ny: {}",
					f64_to_float::<T>(closest_point.0).human_display(false),
					f64_to_float::<T>(closest_point.1).human_display(false)
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
			show_popup_label(
				ui,
				Id::new("point_popup"),
				format!(
					"{}\nx:{}\ny: {}",
					hovered_point.name(),
					f64_to_float::<T>(hovered_point.x).human_display(false),
					f64_to_float::<T>(hovered_point.y).human_display(false)
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

fn add_new_entry_btn<T: EvalexprFloat>(
	ui: &mut egui::Ui, next_id: &mut u64, entries: &mut Vec<Entry<T>>, can_add_folder: bool,
) -> bool {
	let mut needs_recompilation = false;

	if entries.len() >= 1000 {
		return false;
	}
	ui.menu_button("➕ Add", |ui| {
		let new_function = Entry::new_function(*next_id, String::new());
		if ui.button(new_function.type_name()).clicked() {
			entries.push(new_function);
			*next_id += 1;
			needs_recompilation = true;
		}
		let new_constant = Entry::new_constant(*next_id);
		if ui.button(new_constant.type_name()).clicked() {
			entries.push(new_constant);
			*next_id += 1;
			needs_recompilation = true;
		}
		let new_points = Entry::new_points(*next_id);
		if ui.button(new_points.type_name()).clicked() {
			entries.push(new_points);
			*next_id += 1;
			needs_recompilation = true;
		}
		let new_label = Entry::new_label(*next_id);
		if ui.button(new_label.type_name()).clicked() {
			entries.push(new_label);
			*next_id += 1;
		}
		if can_add_folder {
			let new_folder = Entry::new_folder(*next_id);
			if ui.button(new_folder.type_name()).clicked() {
				entries.push(new_folder);
				*next_id += 1;
			}
		}
	});

	needs_recompilation
}

fn show_popup_label(ui: &egui::Ui, id: Id, label: String, pos: [f32; 2]) {
	egui::Area::new(id).fixed_pos([pos[0] + 5.0, pos[1] + 5.0]).interactable(false).show(ui.ctx(), |ui| {
		egui::Frame::popup(ui.style()).show(ui, |ui| {
			ui.horizontal(|ui| ui.label(label));
		});
	});
}
