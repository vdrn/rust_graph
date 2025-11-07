use std::io::Write;
use std::path::PathBuf;

use ahash::AHashMap;
use eframe::egui::{self, Grid, Id, Modal};
use evalexpr::{EvalexprFloat, EvalexprNumericTypes};
use serde::{Deserialize, Serialize};

use crate::entry::FunctionStyle;
use crate::{ConstantType, Entry, EntryType, PointEntry, State, UiState};

#[derive(Serialize, Deserialize)]
pub struct EntrySerialized {
	name:    String,
	visible: bool,
	color:   usize,
	value:   EntryValueSerialized,
}
#[derive(Serialize, Deserialize)]
pub enum EntryValueSerialized {
	Function {
		text:             String,
		#[serde(default)]
		ranged:           bool,
		#[serde(default)]
		range_start_text: String,
		#[serde(default)]
		range_end_text:   String,
		#[serde(default)]
		style:            FunctionStyle,
		#[serde(default)]
		multiline:        bool,
	},
	Constant {
		value: f64,
		step:  f64,
		ty:    ConstantType,
	},
	Points(Vec<EntryPointSerialized>),
	Integral {
		func_text:  String,
		lower_text: String,
		upper_text: String,

		#[serde(default)]
		resolution: usize,
		#[serde(default)]
		multiline:  bool,
	},
	Label {
		text_x:    String,
		text_y:    String,
		text_size: String,
		underline: bool,
	},
}
#[derive(Serialize, Deserialize)]
pub struct EntryPointSerialized {
	x: String,
	y: String,
}

pub fn serialize_to<T: EvalexprNumericTypes>(writer: impl Write, entries: &[Entry<T>]) -> std::io::Result<()> {
	let mut result = Vec::new();
	for entry in entries {
		let entry_serialized = EntrySerialized {
			name:    entry.name.clone(),
			visible: entry.visible,
			color:   entry.color,
			value:   match &entry.ty {
				EntryType::Function {
					text,
					ranged,
					range_start_text,
					range_end_text,
					style,
					multiline,
					..
				} => EntryValueSerialized::Function {
					text:             text.clone(),
					ranged:           *ranged,
					range_start_text: range_start_text.clone(),
					range_end_text:   range_end_text.clone(),
					style:            style.clone(),
					multiline:        *multiline,
				},
				EntryType::Constant { value, step, ty } => {
					EntryValueSerialized::Constant { value: value.to_f64(), step: *step, ty: ty.clone() }
				},
				EntryType::Points(points) => {
					let mut points_serialized = Vec::new();
					for point in points {
						let point_serialized =
							EntryPointSerialized { x: point.text_x.clone(), y: point.text_y.clone() };
						points_serialized.push(point_serialized);
					}
					EntryValueSerialized::Points(points_serialized)
				},
				EntryType::Integral { func_text, lower_text, upper_text, resolution, multiline, .. } => {
					EntryValueSerialized::Integral {
						func_text:  func_text.clone(),
						lower_text: lower_text.clone(),
						upper_text: upper_text.clone(),
						resolution: *resolution,
						multiline:  *multiline,
					}
				},
				EntryType::Label { text_x, text_y, text_size, underline, .. } => EntryValueSerialized::Label {
					text_x:    text_x.clone(),
					text_y:    text_y.clone(),
					text_size: text_size.clone(),
					underline: *underline,
				},
			},
		};
		result.push(entry_serialized);
	}
	serde_json::to_writer(writer, &result)?;
	Ok(())
}

#[cfg(target_arch = "wasm32")]
pub fn deserialize_from_url<T: EvalexprNumericTypes>() -> Result<Vec<Entry<T>>, String> {
	let href = web_sys::window()
		.expect("Couldn't get window")
		.document()
		.expect("Couldn't get document")
		.location()
		.expect("Couldn't get location")
		.href()
		.expect("Couldn't get href");

	if !href.contains('#') {
		return Ok(Vec::new());
	}
	let Some(without_prefix) = href.split('#').last() else {
		return Ok(Vec::new());
	};

	let decoded = urlencoding::decode(without_prefix).map_err(|e| e.to_string())?;
	deserialize_from(decoded.as_bytes())
}

pub fn deserialize_from<T: EvalexprNumericTypes>(reader: &[u8]) -> Result<Vec<Entry<T>>, String> {
	let entries: Vec<EntrySerialized> = serde_json::from_slice(reader).map_err(|e| e.to_string())?;
	let mut result = Vec::new();
	for (id, entry) in entries.into_iter().enumerate() {
		let entry_deserialized = Entry {
			id:      id as u64,
			name:    entry.name,
			visible: entry.visible,
			color:   entry.color,
			ty:      match entry.value {
				EntryValueSerialized::Function {
					text,
					ranged,
					range_start_text,
					range_end_text,
					style,
					multiline,
				} => EntryType::Function {
					func: evalexpr::build_operator_tree::<T>(&text).ok(),
					text,
					ranged,
					range_start: evalexpr::build_operator_tree::<T>(&range_start_text).ok(),
					range_end: evalexpr::build_operator_tree::<T>(&range_end_text).ok(),

					range_start_text,
					range_end_text,
					style,
					multiline,
				},
				EntryValueSerialized::Constant { value, step, ty } => {
					EntryType::Constant { value: T::Float::f64_to_float(value), step, ty }
				},
				EntryValueSerialized::Points(points) => {
					let mut points_deserialized = Vec::new();
					for point in points {
						let point_deserialized = PointEntry {
							x:      evalexpr::build_operator_tree::<T>(&point.x).ok(),
							y:      evalexpr::build_operator_tree::<T>(&point.y).ok(),
							text_x: point.x,
							text_y: point.y,
						};
						points_deserialized.push(point_deserialized);
					}
					EntryType::Points(points_deserialized)
				},
				EntryValueSerialized::Integral {
					func_text,
					lower_text,
					upper_text,
					resolution,
					multiline,
				} => EntryType::Integral {
					func: evalexpr::build_operator_tree::<T>(&func_text).ok(),
					lower: evalexpr::build_operator_tree::<T>(&lower_text).ok(),
					upper: evalexpr::build_operator_tree::<T>(&upper_text).ok(),
					func_text,
					lower_text,
					upper_text,
					calculated: None,
					resolution: resolution.max(10),
					multiline,
				},
				EntryValueSerialized::Label { text_x, text_y, text_size, underline } => EntryType::Label {
					x: evalexpr::build_operator_tree::<T>(&text_x).ok(),
					text_x,
					y: evalexpr::build_operator_tree::<T>(&text_y).ok(),
					text_y,
					size: evalexpr::build_operator_tree::<T>(&text_size).ok(),
					text_size,
					underline,
				},
			},
		};
		result.push(entry_deserialized);
	}
	Ok(result)
}
#[cfg(target_arch = "wasm32")]
pub fn save_file<T: EvalexprNumericTypes>(
	ui_state: &mut UiState, state: &State<T>, frame: &mut eframe::Frame,
) {
	let file = format!("{}.json", state.name);
	let mut output = Vec::new();
	if let Err(e) = serialize_to(&mut output, &state.entries) {
		ui_state.serialization_error = Some(e.to_string());
	} else {
		ui_state.serialization_error = None;
		ui_state.web_storage.insert(file, String::from_utf8(output).unwrap());
		if let Some(storage) = frame.storage_mut() {
			storage.flush();
		}
	}
}
#[cfg(not(target_arch = "wasm32"))]
pub fn save_file<T: EvalexprNumericTypes>(
	ui_state: &mut UiState, state: &State<T>, _frame: &mut eframe::Frame,
) {
	use std::path::PathBuf;

	let save_path = PathBuf::from(&ui_state.cur_dir).join(format!("{}.json", state.name));
	if let Some(parent) = save_path.parent() {
		// Recursively create all parent directories if they don't exist

		if std::fs::create_dir_all(parent).is_err() {
			ui_state.serialization_error = Some(format!("Could not create directory: {}", parent.display()));
			return;
		}
	}
	let Ok(mut file) = std::fs::File::create(&save_path) else {
		ui_state.serialization_error = Some(format!("Could not create file: {}", save_path.display()));
		return;
	};
	if let Err(e) = serialize_to(&mut file, &state.entries) {
		ui_state.serialization_error = Some(e.to_string());
	} else {
		ui_state.serialization_error = None;
		load_file_entries(&ui_state.cur_dir, &mut ui_state.web_storage);
	}
}
pub fn load_file_entries(cur_dir: &str, web_st: &mut AHashMap<String, String>) {
	web_st.clear();
	let Ok(entries) = std::fs::read_dir(PathBuf::from(cur_dir)) else {
		// ui.label("No entries found");
		return;
	};
	for entry in entries {
		let Ok(entry) = entry else {
			continue;
		};
		let file_name = entry.file_name();
		let Some(file_name) = file_name.to_str() else {
			continue;
		};
		if !file_name.ends_with(".json") {
			continue;
		}

		web_st.insert(file_name.to_string(), String::new());
	}
}
#[cfg(target_arch = "wasm32")]
pub fn load_file<T: EvalexprNumericTypes>(
	cur_dir: &str, web_st: &AHashMap<String, String>, file_name: &str, state: &mut State<T>,
) -> Result<(), String> {
	if let Some(file) = web_st.get(file_name) {
		let entries = deserialize_from::<T>(file.as_bytes())?;
		state.entries = entries;
		state.name = file_name.strip_suffix(".json").unwrap_or(file_name).to_string();
		state.clear_cache = true;
	}
	Ok(())
}
#[cfg(not(target_arch = "wasm32"))]
pub fn load_file<T: EvalexprNumericTypes>(
	cur_dir: &str, _web_st: &AHashMap<String, String>, file_name: &str, state: &mut State<T>,
) -> Result<(), String> {
	let Ok(file) = std::fs::read(PathBuf::from(cur_dir).join(file_name)) else {
		return Err(format!("Could not open file: {}", file_name));
	};
	let entries = deserialize_from::<T>(&file).map_err(|e| format!("Could not deserialize file: {}", e))?;
	state.entries = entries;
	state.name = file_name.strip_suffix(".json").unwrap_or(file_name).to_string();
	state.clear_cache = true;

	Ok(())
}

pub fn persistence_ui<T: EvalexprNumericTypes>(
	state: &mut State<T>, ui_state: &mut UiState, ui: &mut egui::Ui, frame: &mut eframe::Frame,
) {
	ui.horizontal_top(|ui| {
		ui.label("Name:");
		ui.text_edit_singleline(&mut state.name);
		if !state.name.trim().is_empty() && ui.button("Save").clicked() {
			save_file(ui_state, state, frame);
		}
	});

	#[cfg(not(target_arch = "wasm32"))]
	{
		ui.separator();
		ui.horizontal_top(|ui| {
			ui.label("CWD:");
			let changed = ui.text_edit_singleline(&mut ui_state.cur_dir).changed();

			if ui.button("⟳").clicked() || changed {
				load_file_entries(&ui_state.cur_dir, &mut ui_state.web_storage);
			}
		});
	}

	if !ui_state.web_storage.is_empty() {
		ui.separator();
	}
	ui.horizontal(|ui| {
		Grid::new("files").num_columns(3).striped(true).show(ui, |ui| {
			for file_name in ui_state.web_storage.keys() {
				ui.label(file_name);
				if ui.button("Load").clicked() {
					if let Err(e) = load_file(&ui_state.cur_dir, &ui_state.web_storage, file_name, state) {
						ui_state.serialization_error = Some(format!("Could not open file: {}", e));
						return;
					};
					ui_state.next_id += state.entries.len() as u64;
				}
				if ui.button("Delete").clicked() {
					ui_state.file_to_remove = Some(file_name.clone());
				}
				ui.end_row();
			}
			confirm_remove_dialog(ui, frame, ui_state);
		});
	});
}

fn confirm_remove_dialog(ui: &egui::Ui, _frame: &mut eframe::Frame, ui_state: &mut UiState) {
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
						#[cfg(target_arch = "wasm32")]
						{
							if let Some(storage) = _frame.storage_mut() {
								storage.flush();
							}
							ui_state.web_storage.remove(file);
						}
						#[cfg(not(target_arch = "wasm32"))]
						{
							if let Ok(()) = std::fs::remove_file(PathBuf::from(&ui_state.cur_dir).join(file)) {
								ui_state.web_storage.remove(file);
							}
						}
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
}
