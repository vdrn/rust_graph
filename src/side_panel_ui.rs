use eframe::egui::{
	self, Align, Button, Id, Modal, RichText, ScrollArea, SidePanel, Slider, Stroke, TextEdit, Window
};
use eframe::epaint::Color32;

use evalexpr::{EvalexprFloat, Stack};

use crate::builtins::{init_builtins, show_builtin_information};
use crate::entry::{self, Entry, EntrySymbol, EntryType, prepare_entries};
use crate::graph_ui::IdGenerator;
use crate::persistence::deserialize_graph_state_from_json;
use crate::utils::{duplicate_entry_btn, remove_entry_btn};
use crate::{GraphState, State, UiState, load_graph_state, persistence, scope};

fn perform_pending_action<T: EvalexprFloat>(ui: &mut egui::Ui, ui_state: &mut UiState, state: &mut State<T>) {
	if let Some(action) = ui_state.pending_action.take() {
		match action {
			crate::PendingAction::New => {
				load_graph_state(
					ui_state,
					state,
					Ok(GraphState::new(ui_state.app_config.default_graph_config.clone())),
					None,
				);
			},
			crate::PendingAction::Open => {
				ui_state.showing_open = true;
				#[cfg(not(target_arch = "wasm32"))]
				{
					ui_state.file_dialog.pick_file();
				}
			},
			crate::PendingAction::Close => {
				ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
			},
		}
	}
	ui_state.showing_save_prompt = false;
}

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
	scope!("side_panel");
	let changed = SidePanel::left("left_panel")
		.default_width(200.0)
		.show(ctx, |ui| {
			egui::containers::menu::MenuBar::new().ui(ui, |ui| {
				menu_bar_items(ui, ui_state, state, frame);
			});

			// Graph entries
			ScrollArea::vertical()
				.show(ui, |ui| {
					ui.add_space(10.0);
					ui.horizontal_top(|ui| {
						ui.heading("Rust Graph: ");
						ui.text_edit_singleline(&mut state.graph_state.name);
					});

					ui.separator();

					let mut needs_recompilation = false;

					ui.horizontal_top(|ui| {
						if add_new_entry_btn(
							ui, "New entry", &mut state.graph_state.id_gen, &mut state.graph_state.entries,
							true,
						) {
							needs_recompilation = true;
						}

						if ui.button("❌ Remove all").on_hover_text("Remove all entries").clicked() {
							state.graph_state.entries.clear();
							ui_state.clear_cache = true;
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
					let mut stack = Stack::<T>::new();
					// println!("state.entries: {:?}", state.entries.iter().map(|e|e.id).collect::<Vec<_>>());
					egui_dnd::dnd(ui, "entries_dnd").show_vec(
						&mut state.graph_state.entries,
						|ui, entry, handle, _state| {
							entry_frame(ui, ui_state.app_config.dark_mode, |ui: &mut egui::Ui| {
								let symbol = entry.entry_symbol().symbol(entry.active);
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
													ui, "", &mut state.graph_state.id_gen, entries, false,
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
													ui_state.app_config.dark_mode,
													|ui: &mut egui::Ui| {
														ui.horizontal(|ui| {
															handle.ui(ui, |ui| {
																ui.label("    |")
																	.on_hover_text("Drag to reorder entries");
															});
															ui.horizontal(|ui| {
																let fe_result = entry::entry_ui(
																	ui, &state.ctx, &state.processed_colors,
																	&mut stack, entry, ui_state.clear_cache,
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
											ui, "New Folder Entry", &mut state.graph_state.id_gen, entries,
											false,
										) {
											needs_recompilation = true;
										}
										if let Some(id) = remove_from_folder {
											if let Some(index) = entries.iter().position(|e| e.id == id) {
												entries.remove(index);
											}
										} else if let Some(id) = duplicate_in_folder {
											duplicate_entry(id, &mut state.graph_state.id_gen, entries);
										}
									});
								} else {
									ui.horizontal(|ui| {
										handle.ui(ui, |ui| {
											ui.label("||").on_hover_text("Drag to reorder entries");
										});
										ui.horizontal(|ui| {
											let result = entry::entry_ui(
												ui, &state.ctx, &state.processed_colors, &mut stack, entry,
												ui_state.clear_cache,
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
					if add_new_entry_btn(
						ui, "New Entry", &mut state.graph_state.id_gen, &mut state.graph_state.entries, true,
					) {
						needs_recompilation = true;
					}

					if let Some(id) = remove {
						if let Some(index) = state.graph_state.entries.iter().position(|e| e.id == id) {
							state.graph_state.entries.remove(index);
						}
					} else if let Some(id) = duplicate {
						duplicate_entry(id, &mut state.graph_state.id_gen, &mut state.graph_state.entries);
					}
					ui.separator();

					#[cfg(not(target_arch = "wasm32"))]
					ui.hyperlink_to("View Online", {
						let mut base_url = "https://rust-graph.netlify.app/".to_string();
						base_url.push_str(ui_state.permalink_string.as_str());

						base_url
					});
					ui.separator();

					if let Some(error) = &ui_state.serialization_error {
						if ui.label(RichText::new(error).color(Color32::RED)).clicked() {
							ui_state.serialization_error = None;
						}
					}

					ui.separator();
					ui.hyperlink_to("Github", "https://github.com/vdrn/rust_graph");

					if needs_recompilation || ui_state.clear_cache {
						scope!("prepare_entries");
						match persistence::serialize_to_url(&state.graph_state) {
							Ok(output) => {
								let mut permalink_string = String::with_capacity(output.len() + 1);
								permalink_string.push('#');
								permalink_string.push_str(&output);
								ui_state.permalink_string = permalink_string.clone();
								if ui_state.initial_permalink_string.is_none() {
									ui_state.initial_permalink_string = Some(permalink_string);
								}
								ui_state.scheduled_url_update = true;
							},
							Err(e) => {
								println!("Error: {e}");
							},
						}

						state.ctx.clear();
						init_builtins::<T>(&mut state.ctx);

						prepare_entries(
							&mut state.graph_state.entries, &mut state.ctx, &mut ui_state.prepare_errors,
						);
					} else if animating {
						scope!("prepare_constants");
						state.ctx.clear();
						init_builtins::<T>(&mut state.ctx);
						entry::prepare_constants(
							&mut state.graph_state.entries, &mut state.ctx, &mut ui_state.prepare_errors,
						);
					}
					let changed_any = needs_recompilation || ui_state.clear_cache || animating;
					if changed_any {
						state.processed_colors.clear();
						entry::optimize_entries(
							&mut state.graph_state.entries,
							&mut state.ctx,
							&mut state.processed_colors,
							&mut ui_state.optimization_errors,
						);
						state.processed_colors.sort();
					}

					ui_state.clear_cache = false;

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
pub fn menu_bar_items<T: EvalexprFloat>(
	ui: &mut egui::Ui, ui_state: &mut UiState, state: &mut State<T>, _frame: &mut eframe::Frame,
) {
	// File
	ui.menu_button("📂 File", |ui| {
		// New
		if ui.button("New").clicked() {
			if crate::has_unsaved_changes(ui_state) {
				ui_state.showing_save_prompt = true;
				ui_state.pending_action = Some(crate::PendingAction::New);
			} else {
				load_graph_state(
					ui_state,
					state,
					Ok(GraphState::new(ui_state.app_config.default_graph_config.clone())),
					None,
				);
			}
		}
		if ui.button("Open").clicked() {
			if crate::has_unsaved_changes(ui_state) {
				ui_state.showing_save_prompt = true;
				ui_state.pending_action = Some(crate::PendingAction::Open);
			} else {
				ui_state.showing_open = true;

				#[cfg(not(target_arch = "wasm32"))]
				{
					ui_state.file_dialog.pick_file();
				}
			}
		}
		if ui.add_enabled(ui_state.current_file_path.is_some(), egui::Button::new("Save")).clicked() {
			if let Some(file_path) = &ui_state.current_file_path {
				#[cfg(not(target_arch = "wasm32"))]
				{
					let path = std::path::PathBuf::from(file_path);
					if let Err(e) = crate::persistence::save_file_desktop(path, state) {
						ui_state.serialization_error = Some(e);
					} else {
						ui_state.serialization_error = None;
						// update initial_permalink_string to current
						ui_state.initial_permalink_string = Some(ui_state.permalink_string.clone());
					}
				}
				#[cfg(target_arch = "wasm32")]
				{
					if let Err(e) = crate::persistence::save_file_wasm(ui_state, state, _frame) {
						ui_state.serialization_error = Some(e);
					} else {
						ui_state.serialization_error = None;
						ui_state.initial_permalink_string = Some(ui_state.permalink_string.clone());
					}
				}
			}
		}
		if ui.button("Save as").clicked() {
			ui_state.showing_save = true;
			#[cfg(not(target_arch = "wasm32"))]
			{
				ui_state.file_dialog.config_mut().default_file_name =
					format!("{}.json", state.graph_state.name);

				ui_state.file_dialog.save_file();
			}
		}
		#[cfg(not(target_arch = "wasm32"))]
		{
			ui.separator();
			if ui.button("Exit").clicked() {
				ui_state.quit(ui.ctx());
			}
		}
	});

	if ui_state.showing_open {
		// Open on WASM
		#[cfg(target_arch = "wasm32")]
		{
			let open_modal = Modal::new(Id::new("Open File WASM")).show(ui.ctx(), |ui| {
				ui.set_width(400.0);
				ui.horizontal(|ui| {
					ui.heading("Saved Graphs");
					if ui.button("Close").clicked() {
						ui.close();
					}
				});
				ui.separator();

				ScrollArea::vertical().max_height(400.0).show(ui, |ui| {
					egui::Grid::new("files").num_columns(3).striped(true).show(ui, |ui| {
						for file_name in ui_state.serialized_states.keys() {
							ui.label(file_name);
							if ui.button("Open").clicked() {
								let graph_state_res =
									crate::persistence::load_file_wasm(&ui_state.serialized_states, file_name);

								load_graph_state(ui_state, state, graph_state_res, Some(file_name.clone()));
								ui.close();
								// todo: hack for borrow checker, fix later
								break;
							}
							if ui.button("Delete").clicked() {
								ui_state.file_to_remove = Some(file_name.clone());
							}
							ui.end_row();
						}

						// confirm_remove_dialog
						if let Some(file) = &ui_state.file_to_remove {
							let modal = Modal::new(Id::new("Confirm remove file")).show(ui.ctx(), |ui| {
								ui.set_width(400.0);
								ui.heading(format!("Are you sure you want to delete '{file}'?"));

								ui.add_space(32.0);

								egui::Sides::new().show(
									ui,
									|_ui| {},
									|ui| {
										if ui.button("Yes").clicked() {
											if let Some(storage) = _frame.storage_mut() {
												storage.flush();
											}
											ui_state.serialized_states.remove(file);
											ui.close();
										}

										if ui.button("No").clicked() {
											ui.close();
										}
									},
								);
							});

							if modal.should_close() {
								ui_state.file_to_remove = None;
							}
						}
					})
				});
			});
			if open_modal.should_close() {
				ui_state.showing_open = false;
			}
		}
		// Open on Desktop
		#[cfg(not(target_arch = "wasm32"))]
		{
			ui_state.file_dialog.update(ui.ctx());

			// Check if the user picked a file.
			if let Some(path) = ui_state.file_dialog.take_picked() {
				let graph_state_res = crate::persistence::load_file_desktop(&path);
				load_graph_state(ui_state, state, graph_state_res, Some(path.to_string_lossy().to_string()));

				ui_state.showing_open = false;
			}
		}
	}

	if ui_state.showing_save {
		// Save on WASM
		#[cfg(target_arch = "wasm32")]
		{
			let save_modal = Modal::new(Id::new("Save File File WASM")).show(ui.ctx(), |ui| {
				ui.set_width(400.0);
				ui.heading("Save Graph");

				ui.horizontal(|ui| {
					ui.label("Name:");
					ui.text_edit_singleline(&mut state.graph_state.name);
				});
				ui.separator();

				ui.horizontal(|ui| {
					if ui.add_enabled(!state.graph_state.name.trim().is_empty(), Button::new("Save")).clicked()
					{
						if let Err(e) = crate::persistence::save_file_wasm(ui_state, state, _frame) {
							ui_state.serialization_error = Some(e);
						} else {
							ui_state.serialization_error = None;
							ui_state.current_file_path = Some(format!("{}.json", state.graph_state.name));
							ui_state.initial_permalink_string = Some(ui_state.permalink_string.clone());
							if ui_state.perform_pending_after_save {
								ui_state.perform_pending_after_save = false;
								perform_pending_action(ui, ui_state, state);
							}
						}
						ui.close();
					}

					if ui.button("Cancel").clicked() {
						ui.close();
					}
				});
			});

			if save_modal.should_close() {
				ui_state.showing_save = false;
				if ui_state.perform_pending_after_save {
					ui_state.perform_pending_after_save = false;
					perform_pending_action(ui, ui_state, state);
				}
			}
		}
		// Save on Desktop
		#[cfg(not(target_arch = "wasm32"))]
		{
			ui_state.file_dialog.update(ui.ctx());

			// Check if the user picked a file.
			if let Some(path) = ui_state.file_dialog.take_picked() {
				ui_state.showing_save = false;

				if !path.is_dir()
					&& let Some(file_name) = path.file_name()
					&& !file_name.is_empty()
				{
					if let Err(e) = crate::persistence::save_file_desktop(path.clone(), state) {
						ui_state.serialization_error = Some(e);
					} else {
						ui_state.serialization_error = None;
						ui_state.current_file_path = Some(path.to_string_lossy().to_string());
						ui_state.initial_permalink_string = Some(ui_state.permalink_string.clone());
						if ui_state.perform_pending_after_save {
							ui_state.perform_pending_after_save = false;
							perform_pending_action(ui, ui_state, state);
						}
					}
				} else {
					ui_state.serialization_error = Some("Please provide a valid name".to_string());
				}
			}
		}
	}

	if ui_state.showing_save_prompt {
		let save_prompt = Modal::new(Id::new("Save Prompt")).show(ui.ctx(), |ui| {
			ui.set_width(400.0);
			ui.heading("Unsaved Changes");
			ui.label("You have unsaved changes. Do you want to save them?");

			ui.horizontal(|ui| {
				if ui.button("Save").clicked() {
					if ui_state.current_file_path.is_some() {
						// Save directly
						let file_path = ui_state.current_file_path.as_ref().unwrap();
						#[cfg(not(target_arch = "wasm32"))]
						{
							let path = std::path::PathBuf::from(file_path);
							if let Err(e) = crate::persistence::save_file_desktop(path, state) {
								ui_state.serialization_error = Some(e);
							} else {
								ui_state.serialization_error = None;
								ui_state.initial_permalink_string = Some(ui_state.permalink_string.clone());
								perform_pending_action(ui, ui_state, state);
								ui_state.pending_action = None;
							}
						}
						#[cfg(target_arch = "wasm32")]
						{
							if let Err(e) = crate::persistence::save_file_wasm(ui_state, state, _frame) {
								ui_state.serialization_error = Some(e);
							} else {
								ui_state.serialization_error = None;
								ui_state.initial_permalink_string = Some(ui_state.permalink_string.clone());
								perform_pending_action(ui, ui_state, state);
								ui_state.pending_action = None;
							}
						}
					} else {
						// Open save dialog
						ui_state.showing_save = true;
						ui_state.perform_pending_after_save = true;

						#[cfg(not(target_arch = "wasm32"))]
						{
							ui_state.file_dialog.config_mut().default_file_name =
								format!("{}.json", state.graph_state.name);

							ui_state.file_dialog.save_file();
						}
					}
					ui.close();
				}
				if ui.button("Don't Save").clicked() {
					perform_pending_action(ui, ui_state, state);
					ui_state.pending_action = None;
				}
				if ui.button("Cancel").clicked() {
					ui_state.pending_action = None;
					ui.close();
				}
			});
		});
		if save_prompt.should_close() {
			ui_state.showing_save_prompt = false;
		}
	}

	// Examples
	ui.menu_button("📂 Examples", |ui| {
		for entry in crate::EXAMPLES_DIR.entries() {
			if let Some(entry_file) = entry.as_file() {
				if let Some(entry_name) = entry_file.path().file_stem().and_then(|s| s.to_str()) {
					if ui.button(entry_name).clicked() {
						let mut graph_state_res =
							deserialize_graph_state_from_json::<T>(entry_file.contents())
								.map_err(|e| format!("Could not deserialize file: {}", e));
						if let Ok(ref mut gs) = graph_state_res {
							if gs.name.is_empty() {
								gs.name = entry_name.to_string();
							}
						}
						load_graph_state(ui_state, state, graph_state_res, None);
					}
				}
			}
		}
	});

	// Settings
	if ui.button("⚙ Settings").clicked() {
		ui_state.showing_settings = !ui_state.showing_settings;
	}
	if ui_state.showing_settings {
		Window::new("⚙ Settings").open(&mut ui_state.showing_settings).show(ui.ctx(), |ui| {
			ScrollArea::vertical().show(ui, |ui| {
				ui.separator();
				ui.checkbox(&mut ui_state.app_config.use_f32, "Use f32");

				ui.separator();
				ui.add(Slider::new(&mut ui_state.app_config.resolution, 10..=2000).text("Point Resolution"));

				ui.separator();
				ui.add(Slider::new(&mut ui_state.app_config.ui_scale, 1.0..=3.0).text("Ui Scale"));

				ui.separator();
				#[cfg(not(target_arch = "wasm32"))]
				if ui.button("Toggle Fullscreen").clicked()
					|| ui.input(|i| i.key_pressed(egui::Key::F11))
					|| ui.input(|i| i.modifiers.alt && i.key_pressed(egui::Key::Enter))
				{
					ui_state.app_config.fullscreen = !ui_state.app_config.fullscreen;
					ui.ctx()
						.send_viewport_cmd(egui::ViewportCommand::Fullscreen(ui_state.app_config.fullscreen));
				}
			});
		});
	}

	// Help
	if ui.button(format!("{} Help", if ui_state.showing_help { "📖" } else { "📚" })).clicked() {
		ui_state.showing_help = !ui_state.showing_help;
	}
	if ui_state.showing_help {
		Window::new("📖 Help").open(&mut ui_state.showing_help).show(ui.ctx(), |ui| {
			ScrollArea::vertical().show(ui, |ui| {
				show_builtin_information(ui);
			});
		});
	}

	// Dark mode toggle
	if ui
		.button(if ui_state.app_config.dark_mode { "🌙" } else { "☀" })
		.on_hover_text("Toggle Dark Mode")
		.clicked()
	{
		ui_state.app_config.dark_mode = !ui_state.app_config.dark_mode;
	}
}

fn prepare_copied_entry<T: EvalexprFloat>(entry: &mut Entry<T>, id_gen: &mut IdGenerator) {
	if !entry.name.is_empty() {
		entry.name = format!("{}_c", entry.name);
	}
	entry.id = id_gen.next();
	if let EntryType::Folder { entries } = &mut entry.ty {
		for e in entries {
			prepare_copied_entry(e, id_gen);
		}
	}
}
fn duplicate_entry<T: EvalexprFloat>(entry_id: u64, id_gen: &mut IdGenerator, entries: &mut Vec<Entry<T>>) {
	let Some(entry_i) = entries.iter().position(|e| e.id == entry_id) else {
		return;
	};
	let entry = &entries[entry_i];
	let mut new_entry = entry.clone();

	prepare_copied_entry(&mut new_entry, id_gen);
	entries.insert(entry_i + 1, new_entry);
}
fn add_new_entry_btn<T: EvalexprFloat>(
	ui: &mut egui::Ui, text: &str, id_gen: &mut IdGenerator, entries: &mut Vec<Entry<T>>, can_add_folder: bool,
) -> bool {
	let mut needs_recompilation = false;

	if entries.len() >= 1000 {
		return false;
	}
	ui.menu_button(format!("➕ {text}"), |ui| {
		if ui.button(EntrySymbol::Function.symbol_with_name()).clicked() {
			let new_function = Entry::new_function(id_gen.next(), "");
			entries.push(new_function);
			needs_recompilation = true;
		}
		if ui.button(EntrySymbol::Constant.symbol_with_name()).clicked() {
			let new_constant = Entry::new_constant(id_gen.next());
			entries.push(new_constant);
			needs_recompilation = true;
		}
		if ui.button(EntrySymbol::Points.symbol_with_name()).clicked() {
			let new_points = Entry::new_points(id_gen.next());
			entries.push(new_points);
			needs_recompilation = true;
		}
		if ui.button(EntrySymbol::Color.symbol_with_name()).clicked() {
			let new_color = Entry::new_color(id_gen.next());
			entries.push(new_color);
			needs_recompilation = true;
		}
		if can_add_folder {
			if ui.button(EntrySymbol::Folder.symbol_with_name()).clicked() {
				let new_folder = Entry::new_folder(id_gen.next());
				entries.push(new_folder);
			}
		}
	})
	.response
	.on_hover_text(if can_add_folder { "Add new Entry" } else { "Add new Folder Entry" });

	needs_recompilation
}
