use rustc_hash::FxHashSet;
use std::io::Write;
use std::path::{Path, PathBuf};

use base64::Engine;
use serde::{Deserialize, Serialize};

use evalexpr::{EvalexprFloat, istr, istr_empty};

use crate::color::EntryColor;
use crate::custom_rendering::fan_fill_renderer::FillRule;
use crate::entry::{
	ColorEntry, ColorEntryType, EquationType, Expr, FillAlpha, FunctionType, LabelConfig, LabelPosition, LabelSize, LineStyleConfig, MAX_IMPLICIT_RESOLUTION, MIN_IMPLICIT_RESOLUTION, PointDrag, PointDragType, PointStyle, PointsType, preprocess_ast
};
use crate::graph_ui::IdGenerator;
use crate::graph_ui::graph_config::GraphConfig;
use crate::graph_ui::plot_elements::RawPlotElements;
use crate::{ConstantType, Entry, EntryType, GraphState, PointEntry, State};

pub fn default_true() -> bool { true }

#[derive(Serialize, Deserialize)]
pub struct StateSerialized {
	pub entries:      Vec<EntrySerialized>,
	#[serde(default)]
	pub graph_config: GraphConfig,
	#[serde(default)]
	pub id_generator: IdGenerator,
	#[serde(default)]
	pub name:         String,
}

#[derive(Serialize, Deserialize)]
pub struct EntrySerialized {
	id:      u64,
	#[serde(default)]
	name:    String,
	#[serde(default = "default_true")]
	visible: bool,
	#[serde(default)]
	color:   EntryColor,
	ty:      EntryTypeSerialized,
}
#[derive(Clone, Serialize, PartialEq, Deserialize, Default)]
pub struct ExprSer {
	#[serde(default)]
	text:             String,
	#[serde(default)]
	display_rational: bool,
}
impl ExprSer {
	pub fn from_expr<T: EvalexprFloat>(expr: &Expr<T>) -> Self {
		Self { text: expr.text.clone(), display_rational: expr.display_rational }
	}
	pub fn into_expr<T: EvalexprFloat>(self, preprocess: bool) -> Expr<T> {
		let mut equation_type = EquationType::None;
		let mut ast = evalexpr::build_ast::<T>(&self.text).ok();
		if preprocess
			&& let Some((new_ast, new_equation_type)) = ast.take().and_then(|ast| preprocess_ast(ast).ok())
		{
			ast = Some(new_ast);
			equation_type = new_equation_type;
		}
		Expr {
			node: ast.and_then(|ast| evalexpr::build_flat_node_from_ast::<T>(ast).ok()),
			equation_type,

			args: Vec::new(),
			expr_function: None,
			text: self.text,
			display_rational: self.display_rational,
		}
	}
}
#[derive(Serialize, Deserialize)]
pub enum EntryTypeSerialized {
	Function {
		func:                ExprSer,
		#[serde(default)]
		ranged:              bool,
		#[serde(default)]
		range_start:         ExprSer,
		#[serde(default)]
		range_end:           ExprSer,
		#[serde(default)]
		style:               LineStyleConfig,
		#[serde(default)]
		implicit_resolution: usize,
		#[serde(default)]
		parametric_fill:     bool,
		#[serde(default)]
		fill_rule:           FillRule,
		#[serde(default)]
		fill_alpha:          FillAlpha,
	},
	Constant {
		value:       f64,
		step:        f64,
		ty:          ConstantType,
		#[serde(default)]
		range_start: ExprSer,
		#[serde(default)]
		range_end:   ExprSer,
	},
	Points {
		points_ty: EntryPointTypeSerialized,
		#[serde(default)]
		style:     PointStyleSerialized,
	},
	Folder {
		#[serde(default)]
		entries: Vec<EntrySerialized>,
	},
	Color {
		#[serde(default)]
		expr: ExprSer,
		#[serde(default)]
		ty:   ColorEntryType,
	},
}

#[derive(Serialize, Deserialize)]
pub struct EntryPointSerialized {
	#[serde(default)]
	x:         ExprSer,
	#[serde(default)]
	y:         ExprSer,
	#[serde(default)]
	drag_type: PointDragType,
}
#[derive(Serialize, Deserialize)]
pub enum EntryPointTypeSerialized {
	Separate { points: Vec<EntryPointSerialized> },
	SingleExpr { expr: ExprSer },
}

#[derive(Default, Serialize, Deserialize)]
pub struct LabelConfigSerialized {
	#[serde(default)]
	text:   String,
	#[serde(default)]
	size:   LabelSize,
	#[serde(default)]
	pos:    LabelPosition,
	#[serde(default)]
	italic: bool,
	#[serde(default)]
	angle:  ExprSer,
}
impl LabelConfigSerialized {
	fn from_label_config<T: EvalexprFloat>(label_config: &LabelConfig<T>) -> Self {
		Self {
			text:   label_config.text.clone(),
			size:   label_config.size,
			pos:    label_config.pos,
			italic: label_config.italic,
			angle:  ExprSer::from_expr(&label_config.angle),
		}
	}
	fn into_label_config<T: EvalexprFloat>(self) -> LabelConfig<T> {
		LabelConfig {
			text:   self.text,
			size:   self.size,
			pos:    self.pos,
			italic: self.italic,
			angle:  self.angle.into_expr(false),
		}
	}
}
#[derive(Serialize, Default, Deserialize)]
pub struct PointStyleSerialized {
	#[serde(default)]
	show_lines:             bool,
	#[serde(default)]
	show_arrows:            bool,
	#[serde(default)]
	show_points:            bool,
	#[serde(default)]
	line_style:             LineStyleConfig,
	#[serde(default)]
	label_config:           Option<LabelConfigSerialized>,
	#[serde(default)]
	fill:                   bool,
	#[serde(default)]
	fill_rule:              FillRule,
	#[serde(default)]
	connect_first_and_last: bool,
	#[serde(default)]
	fill_alpha:             FillAlpha,
}
impl PointStyleSerialized {
	fn from_point_style<T: EvalexprFloat>(point_style: &PointStyle<T>) -> Self {
		Self {
			show_lines:             point_style.show_lines,
			show_arrows:            point_style.show_arrows,
			show_points:            point_style.show_points,
			line_style:             point_style.line_style.clone(),
			label_config:           point_style
				.label_config
				.as_ref()
				.map(LabelConfigSerialized::from_label_config),
			fill:                   point_style.fill,
			fill_rule:              point_style.fill_rule,
			fill_alpha:             point_style.fill_alpha,
			connect_first_and_last: point_style.connect_first_and_last,
		}
	}
	fn into_point_style<T: EvalexprFloat>(self) -> PointStyle<T> {
		PointStyle {
			show_lines:             self.show_lines,
			show_arrows:            self.show_arrows,
			show_points:            self.show_points,
			line_style:             self.line_style,
			label_config:           self.label_config.map(|label_config| label_config.into_label_config()),
			fill:                   self.fill,
			fill_rule:              self.fill_rule,
			fill_alpha:             self.fill_alpha,
			connect_first_and_last: self.connect_first_and_last,
		}
	}
}

pub fn serialize_entries<T: EvalexprFloat>(entries: &[Entry<T>]) -> Vec<EntrySerialized> {
	let mut result = Vec::with_capacity(entries.len());
	for entry in entries {
		let entry_serialized = EntrySerialized {
			id:      entry.id,
			name:    entry.name.clone(),
			visible: entry.active,
			color:   entry.color.clone(),
			ty:      match &entry.ty {
				EntryType::Function {
					func,
					range_start,
					range_end,
					style,
					parametric,
					implicit_resolution,
					parametric_fill,
					fill_rule,
					fill_alpha,
					..
				} => EntryTypeSerialized::Function {
					func:                ExprSer::from_expr(func),
					ranged:              *parametric,
					range_start:         ExprSer::from_expr(range_start),
					range_end:           ExprSer::from_expr(range_end),
					style:               style.clone(),
					fill_rule:           *fill_rule,
					implicit_resolution: *implicit_resolution,
					parametric_fill:     *parametric_fill,
					fill_alpha:          *fill_alpha,
				},
				EntryType::Constant { value, step, ty, istr_name: _, range_start, range_end } => {
					EntryTypeSerialized::Constant {
						value:       value.to_f64(),
						step:        *step,
						ty:          ty.clone(),
						range_start: ExprSer::from_expr(range_start),
						range_end:   ExprSer::from_expr(range_end),
					}
				},
				EntryType::Points { points_ty, style, identifier: _ } => match points_ty {
					PointsType::Separate(points) => {
						let mut points_serialized = Vec::new();
						for point in points {
							let point_serialized = EntryPointSerialized {
								x:         ExprSer::from_expr(&point.x),
								y:         ExprSer::from_expr(&point.y),
								drag_type: point.drag.drag_type,
							};
							points_serialized.push(point_serialized);
						}
						EntryTypeSerialized::Points {
							points_ty: EntryPointTypeSerialized::Separate { points: points_serialized },
							// points:    Vec::new(),
							style:     PointStyleSerialized::from_point_style(style),
						}
					},
					PointsType::SingleExpr { expr, .. } => EntryTypeSerialized::Points {
						points_ty: EntryPointTypeSerialized::SingleExpr { expr: ExprSer::from_expr(expr) },
						// points:    Vec::new(),
						style:     PointStyleSerialized::from_point_style(style),
					},
				},
				EntryType::Folder { entries } => {
					let ser_entries = serialize_entries(entries);
					EntryTypeSerialized::Folder { entries: ser_entries }
				},
				EntryType::Color(color) => {
					EntryTypeSerialized::Color { expr: ExprSer::from_expr(&color.expr), ty: color.ty }
				},
			},
		};
		result.push(entry_serialized);
	}
	result
}
pub fn serialize_graph_state<T: EvalexprFloat>(graph_state: &GraphState<T>) -> StateSerialized {
	let state_serialized = StateSerialized {
		entries:      serialize_entries(&graph_state.entries),
		graph_config: graph_state.current_graph_config.clone(),
		id_generator: graph_state.id_gen.clone(),
		name:         graph_state.name.clone(),
	};

	state_serialized
}
pub fn serialize_graph_state_to_json<T: EvalexprFloat>(
	writer: impl Write, graph_state: &GraphState<T>,
) -> Result<(), String> {
	let ser = serialize_graph_state(graph_state);
	serde_json::to_writer(writer, &ser).map_err(|e| e.to_string())?;
	Ok(())
}
pub fn serialize_to_url<T: EvalexprFloat>(graph_state: &GraphState<T>) -> Result<String, String> {
	let ser = serialize_graph_state(graph_state);
	let bincoded =
		bincode::serde::encode_to_vec(&ser, bincode::config::standard()).map_err(|e| e.to_string())?;
	let base64_encoded = base64::engine::general_purpose::STANDARD.encode(bincoded);
	Ok(base64_encoded)
}

#[cfg(target_arch = "wasm32")]
pub fn deserialize_from_url<T: EvalexprFloat>(
) -> Result<Option<GraphState<T>>, String> {
	let href = web_sys::window()
		.expect("Couldn't get window")
		.document()
		.expect("Couldn't get document")
		.location()
		.expect("Couldn't get location")
		.href()
		.expect("Couldn't get href");
	// let href = "";

	if !href.contains('#') {
		return Ok(None);
	}
	let Some(without_prefix) = href.split('#').last() else {
		return Ok(None);
	};

	let base64_decoded =
		base64::engine::general_purpose::STANDARD.decode(without_prefix).map_err(|e| e.to_string())?;
	let entries_ser = bincode::serde::decode_from_slice(&base64_decoded, bincode::config::standard())
		.map_err(|e| e.to_string())?
		.0;
	Ok(Some(deserialize_graph_state(entries_ser)?))
}

pub fn deserialize_graph_state<T: EvalexprFloat>(ser: StateSerialized) -> Result<GraphState<T>, String> {
	let entries = deserialize_entries::<T>(ser.entries);
	let mut unique_ids = FxHashSet::default();
	let mut max_id = 0;
	for entry in entries.iter() {
		max_id = entry.id.max(max_id);
		if !unique_ids.insert(entry.id) {
			return Err(format!("Error deserializing graph state: Duplicate id: {}", entry.id));
		}
		if let EntryType::Folder { entries } = &entry.ty {
			for sub_entry in entries.iter() {
				max_id = sub_entry.id.max(max_id);
				if !unique_ids.insert(sub_entry.id) {
					return Err(format!("Error deserializing graph state: Duplicate id: {}", sub_entry.id));
				}
			}
		}
	}

	let max_id = max_id + 1;

	Ok(GraphState {
		entries,
		saved_graph_config: ser.graph_config.clone(),
		current_graph_config: ser.graph_config,
		name: ser.name,
		id_gen: IdGenerator::new(max_id),
		prev_plot_transform: None,
	})
}
pub fn deserialize_entries<T: EvalexprFloat>(entries: Vec<EntrySerialized>) -> Vec<Entry<T>> {
	let mut result = Vec::new();
	for entry in entries {
		let entry_deserialized = Entry {
			id:                entry.id,
			active:            entry.visible,
			color:             entry.color,
			raw_plot_elements: RawPlotElements::empty(),
			ty:                match entry.ty {
				EntryTypeSerialized::Function {
					func,
					ranged,
					range_start,
					range_end,
					style,
					implicit_resolution,
					fill_rule,
					parametric_fill,
					fill_alpha,
				} => EntryType::Function {
					parametric: ranged,
					parametric_fill,
					fill_rule,
					identifier: istr_empty(),
					fill_alpha,
					func: func.into_expr(true),
					// Actual type will be set in compilation step later
					ty: FunctionType::Expression,

					can_be_drawn: true,
					range_start: range_start.into_expr(false),
					range_end: range_end.into_expr(false),
					implicit_resolution: implicit_resolution
						.clamp(MIN_IMPLICIT_RESOLUTION, MAX_IMPLICIT_RESOLUTION),

					style,
				},
				EntryTypeSerialized::Constant { value, step, ty, range_start, range_end } => {
					EntryType::Constant {
						value: T::from_f64(value),
						step,
						ty,
						istr_name: istr(&entry.name),
						range_start: range_start.into_expr(false),
						range_end: range_end.into_expr(false),
					}
				},
				EntryTypeSerialized::Points { points_ty, style } => match points_ty {
					EntryPointTypeSerialized::Separate { points } => {
						let mut points_deserialized = Vec::new();
						for point in points {
							let point_deserialized = PointEntry {
								x:    point.x.clone().into_expr(false),
								y:    point.y.clone().into_expr(false),
								drag: PointDrag {
									drag_point:               None,
									drag_type:                point.drag_type,
									both_drag_dirs_available: true,
								},
								val:  None,
							};
							points_deserialized.push(point_deserialized);
						}
						EntryType::Points {
							points_ty: PointsType::Separate(points_deserialized),

							style:      PointStyleSerialized::into_point_style(style),
							identifier: istr_empty(),
						}
					},
					EntryPointTypeSerialized::SingleExpr { expr } => EntryType::Points {
						points_ty:  PointsType::SingleExpr { expr: expr.into_expr(false), val: vec![] },
						style:      PointStyleSerialized::into_point_style(style),
						identifier: istr_empty(),
					},
				},
				EntryTypeSerialized::Folder { entries } => {
					let entries = deserialize_entries(entries);
					EntryType::Folder { entries }
				},
				EntryTypeSerialized::Color { expr, ty } => {
					EntryType::Color(ColorEntry { expr: expr.into_expr(false), ty })
				},
			},
			name:              entry.name,
		};

		result.push(entry_deserialized);
	}
	result
}

pub fn deserialize_graph_state_from_json<T: EvalexprFloat>(reader: &[u8]) -> Result<GraphState<T>, String> {
	let graph_state: StateSerialized = serde_json::from_slice(reader).map_err(|e| e.to_string())?;
	deserialize_graph_state(graph_state)
}

#[cfg(target_arch = "wasm32")]
pub fn save_file_wasm<T: EvalexprFloat>(
	ui_state: &mut crate::UiState, state: &State<T>, frame: &mut eframe::Frame,
) -> Result<(), String> {
	let file = format!("{}.json", state.graph_state.name);
	let mut output = Vec::new();
	if let Err(e) = serialize_graph_state_to_json(&mut output, &state.graph_state) {
		return Err(e.to_string());
	}
	ui_state.serialized_states.insert(file, String::from_utf8(output).unwrap());
	if let Some(storage) = frame.storage_mut() {
		storage.flush();
	}
	Ok(())
}
#[cfg(not(target_arch = "wasm32"))]
pub fn save_file_desktop<T: EvalexprFloat>(save_path: PathBuf, state: &State<T>) -> Result<(), String> {
	if let Some(parent) = save_path.parent() {
		// Recursively create all parent directories if they don't exist
		if std::fs::create_dir_all(parent).is_err() {
			return Err(format!("Could not create directory: {}", parent.display()));
		}
	}
	let Ok(mut file) = std::fs::File::create(&save_path) else {
		return Err(format!("Could not create file: {}", save_path.display()));
	};
	serialize_graph_state_to_json(&mut file, &state.graph_state)
}

#[cfg(target_arch = "wasm32")]
pub fn load_file_wasm<T: EvalexprFloat>(
	ser_states: &std::collections::BTreeMap<String, String>, file_name: &str,
) -> Result<GraphState<T>, String> {
	if let Some(file) = ser_states.get(file_name) {
		let mut graph_state = deserialize_graph_state_from_json::<T>(file.as_bytes())?;
		if graph_state.name.is_empty() {
			graph_state.name = file_name.strip_suffix(".json").unwrap_or(file_name).to_string();
		}
		Ok(graph_state)
	} else {
		Err(format!("Could not find file: {}", file_name))
	}
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_file_desktop<T: EvalexprFloat>(path: &Path) -> Result<GraphState<T>, String> {
	let Ok(file) = std::fs::read(path) else {
		return Err(format!("Could not open file: {}", path.display()));
	};
	let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
	let mut graph_state = deserialize_graph_state_from_json::<T>(&file)
		.map_err(|e| format!("Could not deserialize file: {}", e))?;
	if graph_state.name.is_empty() {
		graph_state.name = name;
	}
	Ok(graph_state)
}
