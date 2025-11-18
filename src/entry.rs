use alloc::sync::Arc;
use core::cell::{Cell, RefMut};
use core::mem;
use core::ops::RangeInclusive;
use evalexpr::error::EvalexprResultValue;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::sync::LazyLock;

use eframe::egui::containers::menu::{MenuButton, MenuConfig, SubMenuButton};
use eframe::egui::{
	self, Align, Button, Color32, DragValue, Id, PopupCloseBehavior, RichText, Slider, Stroke, TextEdit, Widget, vec2
};
use egui_plot::{Line, PlotPoint, PlotPoints, Points, Polygon, Text};
use evalexpr::{
	EvalexprError, EvalexprFloat, EvalexprResult, ExpressionFunction, FlatNode, Function, HashMapContext, IStr, Stack, Value, istr
};
use regex::Regex;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use thread_local::ThreadLocal;

use crate::{ReservedVars, ThreadLocalContext, marching_squares, thread_local_get};

pub const DEFAULT_IMPLICIT_RESOLUTION: usize = 200;
pub const MAX_IMPLICIT_RESOLUTION: usize = 500;
pub const MIN_IMPLICIT_RESOLUTION: usize = 10;

pub struct DrawBuffer {
	pub lines:    Vec<DrawLine>,
	pub points:   Vec<DrawPoint>,
	pub polygons: Vec<DrawPolygonGroup>,
	pub texts:    Vec<DrawText>,
}
#[allow(clippy::non_send_fields_in_send_ty)]
/// SAFETY: Line/Points/Polygon are not Send/Sync because of `ExplicitGenerator` callbacks.
/// We dont use those so we're fine.
unsafe impl Send for DrawBuffer {}

impl Default for DrawBuffer {
	fn default() -> Self {
		Self {
			lines:    Vec::with_capacity(32),
			points:   Vec::with_capacity(32),
			polygons: Vec::with_capacity(4),
			texts:    Vec::with_capacity(4),
		}
	}
}

pub struct DrawLine {
	pub sorting_index: u32,
	pub line:          egui_plot::Line<'static>,
	pub id:            egui::Id,
	pub width:         f32,
}
impl DrawLine {
	pub fn new(sorting_index: u32, id: Id, width: f32, line: egui_plot::Line<'static>) -> Self {
		Self { sorting_index, id, width, line }
	}
}
#[derive(Clone, Debug)]
pub struct PointInteraction {
	pub ty:     PointInteractionType,
	pub x:      f64,
	pub y:      f64,
	pub radius: f32,
}
impl PointInteraction {
	pub fn name(&self) -> &'static str {
		match self.ty {
			PointInteractionType::Draggable { .. } | PointInteractionType::Other(OtherPointType::Point) => {
				"Point"
			},
			PointInteractionType::Other(OtherPointType::IntersectionWithXAxis) => "Intersection with x axis",
			PointInteractionType::Other(OtherPointType::IntersectionWithYAxis) => "Intersection with y axis",
			PointInteractionType::Other(OtherPointType::Intersection) => "Intersection",
			PointInteractionType::Other(OtherPointType::Minima) => "Minima",
			PointInteractionType::Other(OtherPointType::Maxima) => "Maxima",
		}
	}
}
#[derive(Clone, Debug)]
pub enum OtherPointType {
	Point,
	Minima,
	Maxima,
	Intersection,
	IntersectionWithXAxis,
	IntersectionWithYAxis,
}
#[derive(Clone, Debug)]
pub enum PointInteractionType {
	Draggable { i: (Id, u32) },
	Other(OtherPointType),
}
pub struct DrawPoint {
	pub sorting_index: u64,
	pub interaction:   PointInteraction,
	pub points:        egui_plot::Points<'static>,
}
impl DrawPoint {
	pub fn new(i1: u32, i2: u32, selectable: PointInteraction, points: egui_plot::Points<'static>) -> Self {
		Self { sorting_index: ((i1 as u64) << 32) | i2 as u64, interaction: selectable, points }
	}
}
pub struct DrawPolygonGroup {
	pub sorting_index: u32,
	pub polygons:      Vec<egui_plot::Polygon<'static>>,
}
impl DrawPolygonGroup {
	pub fn new(sorting_index: u32, polygons: Vec<egui_plot::Polygon<'static>>) -> Self {
		Self { sorting_index, polygons }
	}
}
pub struct DrawText {
	pub sorting_index: u32,
	pub text:          egui_plot::Text,
}
impl DrawText {
	pub fn new(sorting_index: u32, text: egui_plot::Text) -> Self { Self { sorting_index, text } }
}

/// SAFETY: Not Sync because of `ExplicitGenerator` callbacks, but we dont use those.
unsafe impl Sync for DrawLine {}

pub const COLORS: &[Color32; 20] = &[
	Color32::from_rgb(255, 107, 107), // Bright coral red
	Color32::from_rgb(78, 205, 196),  // Turquoise
	Color32::from_rgb(69, 183, 209),  // Sky blue
	Color32::from_rgb(255, 160, 122), // Light salmon
	Color32::from_rgb(152, 216, 200), // Mint green
	Color32::from_rgb(255, 217, 61),  // Golden yellow
	Color32::from_rgb(107, 207, 127), // Fresh green
	Color32::from_rgb(199, 125, 255), // Bright purple
	Color32::from_rgb(255, 133, 161), // Pink
	Color32::from_rgb(93, 173, 226),  // Bright blue
	Color32::from_rgb(248, 183, 57),  // Orange
	Color32::from_rgb(127, 219, 255), // Aqua
	Color32::from_rgb(57, 255, 20),   // Neon green
	Color32::from_rgb(255, 20, 147),  // Deep pink
	Color32::from_rgb(0, 217, 255),   // Cyan
	Color32::from_rgb(255, 179, 71),  // Peach
	Color32::from_rgb(139, 92, 246),  // Violet
	Color32::from_rgb(52, 211, 153),  // Emerald
	Color32::from_rgb(244, 114, 182), // Hot pink
	Color32::from_rgb(251, 191, 36),  // Amber
];
pub const NUM_COLORS: usize = COLORS.len();

#[derive(Clone, Debug)]
pub struct Entry<T: EvalexprFloat> {
	pub id:      u64,
	pub name:    String,
	pub visible: bool,
	pub color:   usize,
	pub ty:      EntryType<T>,
}
impl<T: EvalexprFloat> core::hash::Hash for Entry<T> {
	fn hash<H: core::hash::Hasher>(&self, state: &mut H) { self.id.hash(state); }
}

#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize, Default)]
pub enum TextboxType {
	#[default]
	SingleLineClipped,
	SingleLineExpanded,
	MultiLine,
}
impl TextboxType {
	pub fn symbol(self) -> &'static str {
		match self {
			TextboxType::SingleLineClipped => "☐",
			TextboxType::SingleLineExpanded => "↔",
			TextboxType::MultiLine => "↕",
		}
	}
	pub fn name(self) -> &'static str {
		match self {
			TextboxType::SingleLineClipped => "☐ Single line clipped",
			TextboxType::SingleLineExpanded => "↔ Single line expanded",
			TextboxType::MultiLine => "↕ Multi line",
		}
	}
}
#[derive(Clone, Debug)]
pub struct Expr<T: EvalexprFloat> {
	pub text:          String,
	pub node:          Option<FlatNode<T>>,
	pub inlined_node:  Option<FlatNode<T>>,
	pub expr_function: Option<ExpressionFunction<T>>,
	pub textbox_type:  TextboxType,
}

impl<T: EvalexprFloat> Default for Expr<T> {
	fn default() -> Self {
		Self {
			text:          Default::default(),
			node:          Default::default(),
			inlined_node:  Default::default(),
			expr_function: Default::default(),
			textbox_type:  Default::default(),
		}
	}
}
impl<T: EvalexprFloat> Expr<T> {
	fn inline_pass(&mut self, ctx: &mut evalexpr::HashMapContext<T>) -> Result<(), String> {
		if let Some(node) = &self.node {
			self.inlined_node = Some(evalexpr::optimize_flat_node(node, ctx).map_err(|e| e.to_string())?);
		}
		Ok(())
	}
	fn edit_ui(
		&mut self, ui: &mut egui::Ui, hint_text: &str, desired_width: Option<f32>, force_update: bool,
		preprocess: bool,
	) -> Result<bool, String> {
		ui.horizontal_top(|ui| {
			let mut changed = false;
			let original_spacing = ui.style().spacing.item_spacing;
			ui.style_mut().spacing.item_spacing = vec2(0.0, 0.0);
			let mut text_edit = match self.textbox_type {
				TextboxType::SingleLineExpanded => TextEdit::singleline(&mut self.text).clip_text(false),
				TextboxType::SingleLineClipped => TextEdit::singleline(&mut self.text).clip_text(true),
				TextboxType::MultiLine => TextEdit::multiline(&mut self.text).desired_rows(2),
			};

			// let mut layouter = |ui: &egui::Ui, buf: &dyn egui::TextBuffer, wrap_width: f32| {
			// 	let mut layout_job: egui::text::LayoutJob = egui_extras::syntax_highlighting::highlight(
			// 		ui.ctx(),
			// 		ui.style(),
			// 		&CodeTheme::dark(12.0),
			// 		buf.as_str(),
			// 		"Rust",
			// 	);
			// 	layout_job.wrap.max_width = wrap_width;
			// 	ui.fonts_mut(|f| f.layout_job(layout_job))
			// };
			// text_edit = text_edit.layouter(&mut layouter);

			text_edit = text_edit.hint_text(hint_text); //.font(egui::TextStyle::Monospace);
			if let Some(width) = desired_width {
				text_edit = text_edit.desired_width(width);
			}
			if ui.add(text_edit).changed() || force_update {
				if self.text.is_empty() {
					self.node = None;
				} else {
					let temp;
					let mut txt = &self.text;
					if preprocess {
						match preprecess_fn(&self.text) {
							Ok(Some(new_text)) => {
								temp = new_text;
								txt = &temp;
							},
							Err(e) => {
								return Err(e);
							},
							_ => {},
						}
					}

					self.node = match evalexpr::build_operator_tree::<T>(txt) {
						Ok(func) => {
							// println!("func: {:#?}", func);

							Some(func)
						},
						Err(e) => {
							// println!("Error: {e}");
							return Err(e.to_string());
						},
					};
					self.inlined_node = None;
				}

				changed = true;
			}
			// if let Some(postfix) = postfix {
			// 	ui.label(RichText::new(postfix).monospace());
			// }
			ui.menu_button(RichText::new(self.textbox_type.symbol()), |ui| {
				ui.selectable_value(
					&mut self.textbox_type,
					TextboxType::SingleLineClipped,
					TextboxType::SingleLineClipped.name(),
				);
				ui.selectable_value(
					&mut self.textbox_type,
					TextboxType::SingleLineExpanded,
					TextboxType::SingleLineExpanded.name(),
				);
				ui.selectable_value(
					&mut self.textbox_type,
					TextboxType::MultiLine,
					TextboxType::MultiLine.name(),
				);
			});
			ui.style_mut().spacing.item_spacing = original_spacing;
			Ok(changed)
		})
		.inner
	}
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FunctionType {
	X,
	Y,
	Ranged,
	Implicit { complex: bool },
}
#[derive(Clone, Debug)]
pub enum EntryType<T: EvalexprFloat> {
	Function {
		can_be_drawn: bool,
		func:         Expr<T>,
		style:        GLineStyle,
		ty:           FunctionType,

		/// used for `RangedFunction`
		range_start:         Expr<T>,
		/// used for `RangedFunction`
		range_end:           Expr<T>,
		/// used for `ImplicitFunction`
		implicit_resolution: usize,
	},
	Constant {
		istr_name: IStr,
		value:     T,
		step:      f64,
		ty:        ConstantType,
	},
	Points {
		points: Vec<PointEntry<T>>,
		style:  PointStyle,
	},
	Integral {
		func:       Expr<T>,
		lower:      Expr<T>,
		upper:      Expr<T>,
		calculated: Option<T>,
		resolution: usize,
		style:      IntegralStyle,
	},
	Label {
		x:         Expr<T>,
		y:         Expr<T>,
		size:      Expr<T>,
		underline: bool,
	},
	Folder {
		entries: Vec<Entry<T>>,
	},
}

#[derive(Clone, Default, PartialEq, Debug, Serialize, Deserialize)]
pub enum LabelSize {
	#[default]
	Small,
	Medium,
	Large,
}
impl LabelSize {
	pub fn size(&self) -> f32 {
		match self {
			LabelSize::Small => 10.0,
			LabelSize::Medium => 16.0,
			LabelSize::Large => 20.0,
		}
	}
}
#[derive(Clone, Default, PartialEq, Debug, Serialize, Deserialize)]
pub enum LabelPosition {
	#[default]
	Bottom,
	Top,
	Left,
	Right,
	TLeft,
	TRight,
	BLeft,
	BRight,
}
impl LabelPosition {
	fn symbol(&self) -> &'static str {
		match self {
			LabelPosition::Top => "⮉",
			LabelPosition::Bottom => "⮋",
			LabelPosition::Left => "⮈",
			LabelPosition::Right => "⮊",
			LabelPosition::TLeft => "⬉",
			LabelPosition::TRight => "⬈",
			LabelPosition::BLeft => "⬋",
			LabelPosition::BRight => "⬊",
		}
	}
	fn dir(&self) -> egui::Vec2 {
		match self {
			LabelPosition::Top => egui::Vec2::new(0.0, 1.0),
			LabelPosition::Bottom => egui::Vec2::new(0.0, -1.0),
			LabelPosition::Left => egui::Vec2::new(-1.0, 0.0),
			LabelPosition::Right => egui::Vec2::new(1.0, 0.0),
			LabelPosition::TLeft => egui::Vec2::new(-0.71, 0.71),
			LabelPosition::TRight => egui::Vec2::new(0.71, 0.71),
			LabelPosition::BLeft => egui::Vec2::new(-0.71, -0.71),
			LabelPosition::BRight => egui::Vec2::new(0.71, -0.71),
		}
	}
}

#[derive(Clone, Default, PartialEq, Debug, Serialize, Deserialize)]
pub struct LabelConfig {
	#[serde(default)]
	size:   LabelSize,
	#[serde(default)]
	pos:    LabelPosition,
	#[serde(default)]
	italic: bool,
}
impl LabelConfig {
	fn ui(&mut self, ui: &mut egui::Ui) {
		ui.checkbox(&mut self.italic, "Italic");
		ui.horizontal(|ui| {
			SubMenuButton::new(format!("Size {:?}", self.size)).ui(ui, |ui| {
				use LabelSize as LS;
				ui.selectable_value(&mut self.size, LS::Small, RichText::new("Small").size(LS::Small.size()));
				ui.selectable_value(
					&mut self.size,
					LS::Medium,
					RichText::new("Medium").size(LS::Medium.size()),
				);
				ui.selectable_value(&mut self.size, LS::Large, RichText::new("Large").size(LS::Large.size()));
			});
		});
		ui.horizontal(|ui| {
			use LabelPosition as LP;
			SubMenuButton::new(format!("Position {}", self.pos.symbol())).ui(ui, |ui| {
				ui.selectable_value(&mut self.pos, LP::Top, format!("Top {}", LP::Top.symbol()));
				ui.selectable_value(&mut self.pos, LP::Bottom, format!("Bottom {}", LP::Bottom.symbol()));
				ui.selectable_value(&mut self.pos, LP::Left, format!("Left {}", LP::Left.symbol()));
				ui.selectable_value(&mut self.pos, LP::Right, format!("Right {}", LP::Right.symbol()));
				ui.selectable_value(&mut self.pos, LP::TLeft, format!("Top Left {}", LP::TLeft.symbol()));
				ui.selectable_value(&mut self.pos, LP::TRight, format!("Top Right {}", LP::TRight.symbol()));
				ui.selectable_value(&mut self.pos, LP::BLeft, format!("Bottom Left {}", LP::BLeft.symbol()));
				ui.selectable_value(
					&mut self.pos,
					LP::BRight,
					format!("Bottom Right {}", LP::BRight.symbol()),
				);
			});
		});
	}
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct PointStyle {
	show_lines:   bool,
	#[serde(default)]
	show_arrows:  bool,
	show_points:  bool,
	line_style:   GLineStyle,
	#[serde(default)]
	label_config: Option<LabelConfig>,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct IntegralStyle {
	show_function:    bool,
	show_integral_fn: bool,
	show_area:        bool,
}
impl Default for IntegralStyle {
	fn default() -> Self { Self { show_function: false, show_integral_fn: true, show_area: true } }
}

impl Default for PointStyle {
	fn default() -> Self {
		Self {
			show_lines:   true,
			show_points:  true,
			show_arrows:  false,
			label_config: Some(LabelConfig::default()),
			line_style:   GLineStyle::default(),
		}
	}
}

#[derive(Clone, Default, PartialEq, Debug, Serialize, Deserialize)]
pub enum LineStyleType {
	#[default]
	Solid,
	Dotted,
	Dashed,
}
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct GLineStyle {
	line_width:      f32,
	#[serde(default)]
	line_style:      LineStyleType,
	line_style_size: f32,
}
impl Default for GLineStyle {
	fn default() -> Self { Self { line_width: 1.5, line_style: LineStyleType::Solid, line_style_size: 5.5 } }
}
impl GLineStyle {
	fn egui_line_style(&self) -> egui_plot::LineStyle {
		match self.line_style {
			LineStyleType::Solid => egui_plot::LineStyle::Solid,
			LineStyleType::Dotted => egui_plot::LineStyle::Dotted { spacing: self.line_style_size },
			LineStyleType::Dashed => egui_plot::LineStyle::Dashed { length: self.line_style_size },
		}
	}
	fn ui(&mut self, ui: &mut egui::Ui) {
		Slider::new(&mut self.line_width, 0.1..=10.0).text("Line Width").ui(ui);
		ui.separator();
		ui.horizontal(|ui| {
			ui.selectable_value(&mut self.line_style, LineStyleType::Solid, "Solid");
			ui.selectable_value(&mut self.line_style, LineStyleType::Dotted, "Dotted");
			ui.selectable_value(&mut self.line_style, LineStyleType::Dashed, "Dashed");
		});
		match &mut self.line_style {
			LineStyleType::Solid => {},
			LineStyleType::Dotted => {
				ui.add(Slider::new(&mut self.line_style_size, 0.1..=20.0).text("Spacing"));
			},
			LineStyleType::Dashed => {
				ui.add(Slider::new(&mut self.line_style_size, 0.1..=20.0).text("Length"));
			},
		}
	}
}

impl<T: EvalexprFloat> Entry<T> {
	pub fn symbol(&self) -> &'static str {
		match self.ty {
			EntryType::Function { .. } => "λ",
			EntryType::Constant { .. } => {
				if self.visible {
					"⏸"
				} else {
					"⏵"
				}
			},
			EntryType::Points { .. } => "◊",
			EntryType::Integral { .. } => "∫",
			EntryType::Label { .. } => "📃",
			EntryType::Folder { .. } => "📂",
		}
	}
	pub fn type_name(&self) -> &'static str {
		match self.ty {
			EntryType::Function { .. } => "λ   Function",
			EntryType::Constant { .. } => "⏵ Constant",
			EntryType::Points { .. } => "◊ Points",
			EntryType::Integral { .. } => "∫   Integral",
			EntryType::Label { .. } => "📃 Label",
			EntryType::Folder { .. } => "📂 Folder",
		}
	}
	pub fn color(&self) -> Color32 { COLORS[self.color % NUM_COLORS] }
	pub fn new_function(id: u64, text: String) -> Self {
		Self {
			id,
			color: id as usize % NUM_COLORS,
			visible: true,
			name: String::new(),
			ty: EntryType::Function {
				can_be_drawn:        true,
				func:                Expr {
					node: evalexpr::build_operator_tree::<T>(&text).ok(),
					inlined_node: None,
					expr_function: None,
					text,
					textbox_type: TextboxType::default(),
				},
				range_start:         Expr::default(),
				range_end:           Expr::default(),
				ty:                  FunctionType::X,
				style:               GLineStyle::default(),
				implicit_resolution: DEFAULT_IMPLICIT_RESOLUTION,
			},
		}
	}
	pub fn new_constant(id: u64) -> Self {
		Self {
			id,
			color: id as usize % NUM_COLORS,
			visible: false,
			name: String::new(),
			ty: EntryType::Constant {
				istr_name: istr(""),
				value:     T::ZERO,
				step:      0.01,
				ty:        ConstantType::LoopForwardAndBackward {
					start:   -10.0,
					end:     10.0,
					forward: true,
				},
			},
		}
	}
	pub fn new_points(id: u64) -> Self {
		Self {
			id,
			color: id as usize % NUM_COLORS,
			visible: true,
			name: String::new(),
			ty: EntryType::Points { points: vec![PointEntry::default()], style: PointStyle::default() },
		}
	}
	pub fn new_integral(id: u64) -> Self {
		Self {
			id,
			color: id as usize % NUM_COLORS,
			visible: true,
			name: String::new(),
			ty: EntryType::Integral {
				func:  Expr::default(),
				lower: Expr::default(),
				upper: Expr::default(),

				calculated: None,
				resolution: 500,
				style:      IntegralStyle::default(),
			},
		}
	}
	pub fn new_label(id: u64) -> Self {
		Self {
			id,
			color: id as usize % NUM_COLORS,
			visible: true,
			name: String::new(),
			ty: EntryType::Label {
				x:         Expr::default(),
				y:         Expr::default(),
				size:      Expr::default(),
				underline: false,
			},
		}
	}
	pub fn new_folder(id: u64) -> Self {
		Self {
			id,
			color: id as usize % NUM_COLORS,
			visible: true,
			name: String::new(),
			ty: EntryType::Folder { entries: Vec::new() },
		}
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub enum PointDragType {
	#[default]
	Both,
	X,
	Y,
	NoDrag,
}
impl PointDragType {
	pub fn symbol(self) -> &'static str {
		match self {
			PointDragType::Both => "↗",
			PointDragType::X => "↔",
			PointDragType::Y => "↕",
			PointDragType::NoDrag => "○",
		}
	}
	fn name(self) -> &'static str {
		match self {
			PointDragType::Both => "↗ Both",
			PointDragType::X => "↔ X",
			PointDragType::Y => "↕ Y",
			PointDragType::NoDrag => "○ No Drag",
		}
	}
}
#[derive(Clone, Debug)]
pub struct PointEntry<T: EvalexprFloat> {
	pub x:                        Expr<T>,
	pub y:                        Expr<T>,
	pub drag_point:               Option<DragPoint>,
	pub drag_type:                PointDragType,
	pub both_drag_dirs_available: bool,
	pub val:                      Option<(T, T)>,
}

impl<T: EvalexprFloat> Default for PointEntry<T> {
	fn default() -> Self {
		Self {
			x:                        Expr::default(),
			y:                        Expr::default(),
			drag_point:               None,
			drag_type:                PointDragType::default(),
			both_drag_dirs_available: true,
			val:                      None,
		}
	}
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConstantType {
	LoopForwardAndBackward { start: f64, end: f64, forward: bool },
	LoopForward { start: f64, end: f64 },
	PlayOnce { start: f64, end: f64 },
	PlayIndefinitely { start: f64 },
}
impl ConstantType {
	pub fn range(&self) -> RangeInclusive<f64> {
		match self {
			ConstantType::LoopForwardAndBackward { start, end, .. } => *start..=*end,
			ConstantType::LoopForward { start, end } => *start..=*end,
			ConstantType::PlayOnce { start, end } => *start..=*end,
			ConstantType::PlayIndefinitely { start } => *start..=f64::INFINITY,
		}
	}
	pub fn symbol(&self) -> &'static str {
		match self {
			ConstantType::LoopForwardAndBackward { .. } => "🔁",
			ConstantType::LoopForward { .. } => "🔂",
			ConstantType::PlayOnce { .. } => "⏯",
			ConstantType::PlayIndefinitely { .. } => "🔀",
		}
	}
	pub fn name(&self) -> &'static str {
		match self {
			ConstantType::LoopForwardAndBackward { .. } => "🔁 Loop forward and Backward",
			ConstantType::LoopForward { .. } => "🔂 Loop forward",
			ConstantType::PlayOnce { .. } => "⏭ Play once",
			ConstantType::PlayIndefinitely { .. } => "🔀 Play indefinitely",
		}
	}
}

pub const STEP_Y_NAME: &str = "___DERIV_STEP__X__";
pub const STEP_X_NAME: &str = "___DERIV_STEP__Y__";
static RESERVED_NAMES: [&str; 7] = ["x", "y", "zx", "zy", "d", STEP_X_NAME, STEP_Y_NAME];
pub struct EditEntryResult {
	pub needs_recompilation: bool,
	pub animating:           bool,
	pub remove:              bool,
	pub error:               Option<String>,
	pub parsed:              bool,
}
pub fn edit_entry_ui<T: EvalexprFloat>(
	ui: &mut egui::Ui, entry: &mut Entry<T>, clear_cache: bool,
) -> EditEntryResult {
	let mut result = EditEntryResult {
		needs_recompilation: false,
		animating:           false,
		remove:              false,
		parsed:              false,

		error: None,
	};

	let (text_col, fill_col) = if entry.visible {
		(Color32::BLACK, entry.color())
	} else {
		(Color32::LIGHT_GRAY, egui::Color32::TRANSPARENT)
	};

	ui.with_layout(egui::Layout::right_to_left(Align::LEFT), |ui| {
		if ui.button("X").clicked() {
			result.remove = true;
			result.needs_recompilation = true;
		}

		let mut style_button = MenuButton::new(RichText::new("🎨").color(Color32::BLACK))
			.config(MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside));
		style_button.button = style_button.button.fill(entry.color());
		style_button.ui(ui, |ui| {
			let mut color_button = SubMenuButton::new(RichText::new("Color").color(Color32::BLACK));
			color_button.button = color_button.button.fill(entry.color());

			color_button.ui(ui, |ui| {
				for i in 0..COLORS.len() {
					if ui.button(RichText::new("     ").background_color(COLORS[i])).clicked() {
						entry.color = i;
					}
				}
			});
			match &mut entry.ty {
				EntryType::Function { style, .. } => {
					ui.separator();

					style.ui(ui);
					ui.separator();
					egui::Sides::new().show(
						ui,
						|_ui| {},
						|ui| {
							if ui.button("Close").clicked() {
								ui.close();
							}
							if ui.button("Reset").clicked() {
								*style = Default::default();
								ui.close();
							}
						},
					);
				},
				EntryType::Points { style, .. } => {
					ui.separator();
					let mut show_label = style.label_config.is_some();
					if ui.checkbox(&mut show_label, "Show label").changed() {
						if show_label {
							style.label_config = Some(LabelConfig::default());
						} else {
							style.label_config = None;
						}
					}
					if let Some(label_config) = &mut style.label_config {
						label_config.ui(ui);
					}
					ui.separator();
					ui.checkbox(&mut style.show_lines, "Show Lines");
					if style.show_lines {
						ui.label("Line Style:");
						ui.checkbox(&mut style.show_arrows, "Show Arrows");
						style.line_style.ui(ui);
					} else {
						style.show_arrows = false;
					}
					ui.separator();
					ui.checkbox(&mut style.show_points, "Show Not Draggable Points");

					egui::Sides::new().show(
						ui,
						|_ui| {},
						|ui| {
							if ui.button("Close").clicked() {
								ui.close();
							}
							if ui.button("Reset").clicked() {
								*style = Default::default();
								ui.close();
							}
						},
					);
				},
				EntryType::Integral { style, .. } => {
					ui.checkbox(&mut style.show_function, "Show Function being integrated");
					if style.show_function {
						ui.checkbox(&mut style.show_area, "Show Area");
					}
					ui.separator();
					ui.checkbox(&mut style.show_integral_fn, "Show Integral Function");
				},
				EntryType::Constant { .. } => {},
				EntryType::Label { .. } => {},
				EntryType::Folder { .. } => {},
			}
		});
		ui.with_layout(egui::Layout::left_to_right(Align::LEFT), |ui| {
			let prev_visible = entry.visible;
			if ui
				.add(
					Button::new(RichText::new(entry.symbol()).strong().monospace().color(text_col))
						.fill(fill_col)
						.corner_radius(10),
				)
				.clicked()
			{
				entry.visible = !entry.visible;
			}

			let name_was_ok = !RESERVED_NAMES.contains(&entry.name.trim());
			if ui.add(TextEdit::singleline(&mut entry.name).desired_width(30.0).hint_text("name")).changed() {
				result.needs_recompilation = true;
			}
			if RESERVED_NAMES.contains(&entry.name.trim()) {
				result.error = Some(format!("{} is reserved name.", entry.name));
				// return;
			} else if !name_was_ok {
				result.parsed = true;
			}
			let color = entry.color();
			match &mut entry.ty {
				EntryType::Folder { .. } => {
					// handled in outer scope
				},
				EntryType::Function { func, range_start, range_end, ty, implicit_resolution, .. } => {
					ui.vertical(|ui| {
						match func.edit_ui(ui, "sin(x)", None, clear_cache, true) {
							Ok(changed) => {
								result.parsed |= changed;
								result.needs_recompilation |= changed;
							},
							Err(e) => {
								result.error = Some(format!("Parsing error: {e}"));
							},
						}

						ui.horizontal(|ui| {
							match ty {
								FunctionType::X | FunctionType::Ranged => {
									let mut is_ranged = *ty == FunctionType::Ranged;
									ui.checkbox(&mut is_ranged, "Parametric");
									*ty = if is_ranged { FunctionType::Ranged } else { FunctionType::X };
								},

								_ => {},
							}
							match ty {
								FunctionType::Ranged => {
									ui.label("Start:");
									match range_start.edit_ui(ui, "", Some(30.0), clear_cache, false) {
										Ok(changed) => {
											result.needs_recompilation |= changed;
											result.parsed |= changed;
										},
										Err(e) => {
											result.error = Some(format!("Parsing error: {e}"));
										},
									}
									ui.label("End:");
									match range_end.edit_ui(ui, "", Some(30.0), clear_cache, false) {
										Ok(changed) => {
											result.needs_recompilation |= changed;
											result.parsed |= changed;
										},
										Err(e) => {
											result.error = Some(format!("Parsing error: {e}"));
										},
									}
								},
								FunctionType::Implicit { complex } => {
									ui.horizontal(|ui| {
										Slider::new(
											implicit_resolution,
											MIN_IMPLICIT_RESOLUTION..=MAX_IMPLICIT_RESOLUTION,
										)
										.text("Implicit Resolution")
										.ui(ui);
										if *complex {
											ui.label("Complex");
										}
									});
								},
								_ => {},
							}
						});
						if let FunctionType::Ranged = ty {
							ui.label("Parametric fns can return 1 or 2 values: f(x)->y  or f(x)->(x,y)");
						}
					});
				},
				EntryType::Integral { func, lower, upper, calculated, resolution, .. } => {
					ui.vertical(|ui| {
						ui.horizontal(|ui| {
							ui.label("Lower:");
							match lower.edit_ui(ui, "lower", Some(50.0), clear_cache, false) {
								Ok(changed) => {
									result.parsed |= changed;
									// needs_recompilation |= changed;
								},
								Err(e) => {
									result.error = Some(format!("Parsing error: {e}"));
								},
							}
							ui.label("Upper:");
							match upper.edit_ui(ui, "upper", Some(50.0), clear_cache, false) {
								Ok(changed) => {
									result.parsed |= changed;
									// needs_recompilation |= changed;
								},
								Err(e) => {
									result.error = Some(format!("Parsing error: {e}"));
								},
							}
						});
						ui.horizontal(|ui| {
							match func.edit_ui(ui, "func", None, clear_cache, false) {
								Ok(changed) => {
									result.parsed |= changed;
									result.needs_recompilation |= changed;
								},
								Err(e) => {
									result.error = Some(format!("Parsing error: {e}"));
								},
							};
							ui.label("dx");
						});
						if let Some(calculated) = calculated {
							ui.label(RichText::new(format!("Value: {}", calculated)).color(color));
						}

						ui.add(Slider::new(resolution, 10..=1000).text("Resolution"));
					});
				},
				EntryType::Label { x, y, size, underline } => {
					ui.horizontal(|ui| {
						match x.edit_ui(ui, "point_x", Some(80.0), clear_cache, false) {
							Ok(changed) => {
								result.parsed |= changed;
								// result.needs_recompilation |= changed;
							},
							Err(e) => {
								result.error = Some(format!("Parsing error: {e}"));
							},
						}
						match y.edit_ui(ui, "point_y", Some(80.0), clear_cache, false) {
							Ok(changed) => {
								result.parsed |= changed;
								// result.needs_recompilation |= changed;
							},
							Err(e) => {
								result.error = Some(format!("Parsing error: {e}"));
							},
						}
						match size.edit_ui(ui, "size", Some(80.0), clear_cache, false) {
							Ok(changed) => {
								result.parsed |= changed;
								// result.needs_recompilation |= changed;
							},
							Err(e) => {
								result.error = Some(format!("Parsing error: {e}"));
							},
						}
						ui.checkbox(underline, "Underline");
					});
				},
				EntryType::Points { points, .. } => {
					let mut remove_point = None;
					ui.vertical(|ui| {
						for (pi, point) in points.iter_mut().enumerate() {
							ui.horizontal(|ui| {
								match point.x.edit_ui(ui, "point_x", Some(80.0), clear_cache, false) {
									Ok(changed) => {
										result.parsed |= changed;
										result.needs_recompilation |= changed;
									},
									Err(e) => {
										result.error = Some(format!("Parsing error: {e}"));
									},
								}
								match point.y.edit_ui(ui, "point_y", Some(80.0), clear_cache, false) {
									Ok(changed) => {
										result.parsed |= changed;
										result.needs_recompilation |= changed;
									},
									Err(e) => {
										result.error = Some(format!("Parsing error: {e}"));
									},
								}
								let mut drag_type_changed = false;
								if !point.both_drag_dirs_available && point.drag_type == PointDragType::Both {
									point.both_drag_dirs_available = false;
									drag_type_changed = true;
								}

								let drag_menu_text = match &point.drag_point {
									Some(d) => match d {
										DragPoint::BothCoordLiterals => {
											format!("{}(x,y)", point.drag_type.symbol())
										},
										DragPoint::XLiteral => format!("{}(x,_)", point.drag_type.symbol()),
										DragPoint::YLiteral => format!("{}(_,y)", point.drag_type.symbol()),
										DragPoint::XConstant(x) => {
											format!("{}({},_)", point.drag_type.symbol(), x)
										},
										DragPoint::YConstant(y) => {
											format!("{}(_, {})", point.drag_type.symbol(), y)
										},
										DragPoint::XLiteralYConstant(y) => {
											format!("{}(x, {})", point.drag_type.symbol(), y)
										},
										DragPoint::YLiteralXConstant(x) => {
											format!("{}({}, y)", point.drag_type.symbol(), x)
										},
										DragPoint::BothCoordConstants(x, y) => {
											format!("{}({}, {})", point.drag_type.symbol(), x, y,)
										},
										DragPoint::SameConstantBothCoords(x) => {
											format!("{}({})", point.drag_type.symbol(), x)
										},
									},
									None => point.drag_type.symbol().to_string(),
								};
								ui.menu_button(drag_menu_text, |ui| {
									drag_type_changed |= ui
										.selectable_value(
											&mut point.drag_type,
											PointDragType::NoDrag,
											PointDragType::NoDrag.name(),
										)
										.changed();
									if point.both_drag_dirs_available {
										drag_type_changed |= ui
											.selectable_value(
												&mut point.drag_type,
												PointDragType::Both,
												PointDragType::Both.name(),
											)
											.changed();
									}
									drag_type_changed |= ui
										.selectable_value(
											&mut point.drag_type,
											PointDragType::X,
											PointDragType::X.name(),
										)
										.changed();
									drag_type_changed |= ui
										.selectable_value(
											&mut point.drag_type,
											PointDragType::Y,
											PointDragType::Y.name(),
										)
										.changed();
								});
								if drag_type_changed {
									result.parsed = true;
									result.needs_recompilation = true;
								}

								if ui.button("❌").clicked() {
									remove_point = Some(pi);
								}
							});
						}
						if let Some(pi) = remove_point {
							points.remove(pi);
						}
						ui.horizontal(|ui| {
							if ui.button("Add Point").clicked() {
								points.push(PointEntry::default());
							}
						});
					});
				},

				EntryType::Constant { value, step, ty, .. } => {
					let original_value = *value;
					let step_f = f64_to_float::<T>(*step);
					let range = ty.range();
					let start = *range.start();
					let end = *range.end();

					ui.vertical(|ui| {
						ui.horizontal(|ui| {
							ui.menu_button(ty.symbol(), |ui| {
								let new_end = if end.is_infinite() { start + 20.0 } else { end };
								let lfab = ConstantType::LoopForwardAndBackward {
									start,
									end: new_end,
									forward: true,
								};
								if ui.button(lfab.name()).clicked() {
									*ty = lfab;
									result.animating = true;
								}
								let lf = ConstantType::LoopForward { start, end: new_end };
								if ui.button(lf.name()).clicked() {
									*ty = lf;
									result.animating = true;
								}
								let po = ConstantType::PlayOnce { start, end: new_end };
								if ui.button(po.name()).clicked() {
									*ty = po;
									result.animating = true;
								}
								let pi = ConstantType::PlayIndefinitely { start };
								if ui.button(pi.name()).clicked() {
									*ty = pi;
									result.animating = true;
								}
							});
							match ty {
								ConstantType::LoopForwardAndBackward { start, end, .. }
								| ConstantType::LoopForward { start, end }
								| ConstantType::PlayOnce { start, end } => {
									DragValue::new(start).prefix("Start:").speed(*step).ui(ui);
									DragValue::new(end).prefix("End:").speed(*step).ui(ui);
									*start = start.min(*end);
									*end = end.max(*start);
								},
								ConstantType::PlayIndefinitely { start } => {
									DragValue::new(start).prefix("Start:").ui(ui);
								},
							}

							DragValue::new(step).prefix("Step:").speed(0.00001).ui(ui);

							if !prev_visible && entry.visible {
								if value.to_f64() >= end {
									*value = f64_to_float::<T>(start);
									result.animating = true;
								}
							}

							if entry.visible {
								ui.ctx().request_repaint();
								result.animating = true;

								match ty {
									ConstantType::LoopForwardAndBackward { forward, .. } => {
										if *forward {
											*value = *value + step_f;
										} else {
											*value = *value - step_f;
										}
										if value.to_f64() > end {
											*forward = false;
											*value = f64_to_float::<T>(end);
										}
										if value.to_f64() < start {
											*forward = true;
											*value = f64_to_float::<T>(start);
										}
									},
									ConstantType::LoopForward { .. } => {
										*value = *value + step_f;
										if value.to_f64() >= end {
											*value = f64_to_float::<T>(start);
										}
									},
									ConstantType::PlayOnce { .. } | ConstantType::PlayIndefinitely { .. } => {
										*value = *value + step_f;
										if value.to_f64() >= end {
											entry.visible = false;
										}
									},
								}
							}
						});

						let mut v = value.to_f64();
						if v > end {
							v = start + (v - end);
							// v -= end - start;
							result.animating = true;
						} else if v < start {
							v = end - (start - v);
							// v += end - start;
							result.animating = true;
						}
						v = v.clamp(start, end);

						if ui
							.add(
								Slider::new(&mut v, range)
									.step_by(*step)
									.clamping(egui::SliderClamping::Never),
							)
							.dragged()
						{
               entry.visible = false;
							// *value = f64_to_float::<T>(v);
							// result.animating = true;
						}
						if original_value.to_f64() != v {
							*value = f64_to_float::<T>(v);
							result.animating = true;
						}
						// *value = f64_to_float::<T>(v);
					});
				},
			}
		});
	});

	result
}

pub fn prepare_entry<T: EvalexprFloat>(
	entry: &mut Entry<T>, ctx: &mut evalexpr::HashMapContext<T>,
) -> Result<(), (u64, String)> {
	if RESERVED_NAMES.contains(&entry.name.trim()) {
		return Ok(());
	}
	match &mut entry.ty {
		EntryType::Folder { entries } => {
			for entry in entries {
				prepare_entry(entry, ctx)?;
			}
		},
		EntryType::Points { points, .. } => {
			for point in points {
				if let (Some(x), Some(y)) = (&point.x.node, &point.y.node) {
					let x_state = analyze_node(x);
					let y_state = analyze_node(y);

					let both_dirs_available = x_state.constants.iter().all(|c| !y_state.constants.contains(c));
					if !both_dirs_available
						&& point.both_drag_dirs_available
						&& !matches!(point.drag_type, PointDragType::NoDrag)
					{
						point.drag_type = PointDragType::X;
					}
					point.both_drag_dirs_available = both_dirs_available;

					match point.drag_type {
						PointDragType::NoDrag => {
							point.drag_point = None;
						},
						PointDragType::Both => {
							if x_state.is_literal && y_state.is_literal {
								point.drag_point = Some(DragPoint::BothCoordLiterals);
							} else if x_state.is_literal
								&& let Some(y_const) = y_state.constants.first()
							{
								point.drag_point = Some(DragPoint::XLiteralYConstant(istr(y_const)));
							} else if y_state.is_literal
								&& let Some(x_const) = x_state.constants.first()
							{
								point.drag_point = Some(DragPoint::YLiteralXConstant(istr(x_const)));
							} else if let (Some(x_const), Some(y_const)) =
								(x_state.constants.first(), y_state.constants.first())
							{
								// todo:
								if x_const == y_const {
									if let Some(y_const) = y_state.constants.get(1) {
										point.drag_point =
											Some(DragPoint::BothCoordConstants(istr(x_const), istr(y_const)));
									} else if let Some(x_const) = x_state.constants.get(1) {
										point.drag_point =
											Some(DragPoint::BothCoordConstants(istr(x_const), istr(y_const)));
									} else {
										point.drag_point = Some(DragPoint::XConstant(istr(x_const)));
									}
								} else {
									point.drag_point =
										Some(DragPoint::BothCoordConstants(istr(x_const), istr(y_const)));
								}
							} else if let Some(x_const) = x_state.constants.first()
							// && y_state.first_constant.is_none()
							{
								point.drag_point = Some(DragPoint::XConstant(istr(x_const)));
							} else if let Some(y_const) = y_state.constants.first() {
								point.drag_point = Some(DragPoint::YConstant(istr(y_const)));
							} else {
								point.drag_point = None;
							}
						},
						PointDragType::X => {
							if x_state.is_literal {
								point.drag_point = Some(DragPoint::XLiteral);
							} else if let Some(x_const) = x_state.constants.first() {
								if x_state.num_identifiers == 1 {
									point.drag_point = Some(DragPoint::XConstant(istr(x_const)));
								} else if y_state.constants.iter().any(|c| c == x_const) {
									point.drag_point = Some(DragPoint::SameConstantBothCoords(istr(x_const)));
								} else {
									point.drag_point = Some(DragPoint::XConstant(istr(x_const)));
								}
							}
						},
						PointDragType::Y => {
							if y_state.is_literal {
								point.drag_point = Some(DragPoint::YLiteral);
							} else if let Some(y_const) = y_state.constants.first() {
								if y_state.num_identifiers == 1 {
									point.drag_point = Some(DragPoint::YConstant(istr(y_const)));
								} else if x_state.constants.iter().any(|c| c == y_const) {
									point.drag_point = Some(DragPoint::SameConstantBothCoords(istr(y_const)));
								} else {
									point.drag_point = Some(DragPoint::YConstant(istr(y_const)));
								}
							}
						},
					}
				} else {
					point.drag_point = None;
				}
			}
		},
		EntryType::Integral { .. } => {},
		EntryType::Label { .. } => {},
		EntryType::Constant { value, istr_name, .. } => {
			*istr_name = istr(entry.name.as_str());
			if !entry.name.is_empty() {
				ctx.set_value(*istr_name, evalexpr::Value::<T>::Float(*value)).unwrap();
			}
		},
		EntryType::Function { func, ty, can_be_drawn, .. } => {
			if let Some(func_node) = func.node.clone() {
				*can_be_drawn = true;

				let mut has_x = false;
				let mut has_y = false;
				let mut has_complex = false;
				for ident in func_node.iter_identifiers() {
					if ident == "x" {
						has_x = true;
					} else if ident == "y" {
						has_y = true;
					} else if ident == "zx" || ident == "zy" {
						has_complex = true;
					}
				}
				if (has_x && has_y) || has_complex {
					if !func.text.contains('=') {
						*can_be_drawn = false;
						// func.node = None;
						// return Err((entry.id, "Implicit function must contain = sign".to_string()));
					}
					*ty = FunctionType::Implicit { complex: has_complex };
				} else if has_y {
					*ty = FunctionType::Y;
				} else if matches!(ty, FunctionType::Implicit { .. } | FunctionType::Y) {
					*ty = FunctionType::X;
				}
			}
		},
	}
	Ok(())
}

pub fn inline_and_fold_entry<T: EvalexprFloat>(
	entry: &mut Entry<T>, ctx: &mut evalexpr::HashMapContext<T>,
	thread_local_context: &Arc<ThreadLocal<ThreadLocalContext<T>>>, reserved_vars: &ReservedVars,
) -> Result<(), (u64, String)> {
	match &mut entry.ty {
		EntryType::Function { func, ty, .. } => {
			if let Err(errs) = func.inline_pass(ctx) {
				return Err((entry.id, errs));
			}
			// println!("inlined_node: {:#?}", func.inlined_node);
			if let Some(inlined_node) = func.inlined_node.clone() {
				let thread_local_context = thread_local_context.clone();
				let ty = *ty;
				let reserved_vars = reserved_vars.clone();
				match ty {
					FunctionType::X => {
						let expr_function = ExpressionFunction::new(inlined_node, vec![istr("x")]);
            // println!("expr_function: {:#?}", expr_function);

						if !entry.name.trim().is_empty() {
							ctx.set_expression_function(istr(entry.name.trim()), expr_function.clone());
						}
						func.expr_function = Some(expr_function);

						return Ok(());
					},
					FunctionType::Ranged | FunctionType::Y => {},
					_ => {},
				}
				if !entry.name.trim().is_empty() {
					let fun = Function::new(move |stack: &mut Stack<T>, context: &HashMapContext<T>| {
						let res = match ty {
							FunctionType::X => {
								unreachable!()
							},
							FunctionType::Ranged | FunctionType::Y => {
								let v = match stack.get_arg(0) {
									Some(Value::Float(x)) => *x,
									Some(Value::Float2(x, _)) => *x,
									_ => T::ZERO,
								};

								if ty == FunctionType::Y {
									eval_with_context_and_y(&inlined_node, stack, context, &reserved_vars, v)
								} else {
									eval_with_context_and_x(&inlined_node, stack, context, &reserved_vars, v)
								}
							},
							FunctionType::Implicit { complex } => {
								if complex {
									let tl_context = thread_local_get(thread_local_context.as_ref());
									let cc_x = tl_context.cc_x.get();
									let cc_y = tl_context.cc_y.get();
									if stack.num_args() == 1 {
										let Ok(v) = stack.get_arg(0).unwrap().as_float2() else {
											return Err(EvalexprError::wrong_function_argument_amount(
												stack.num_args(),
												2,
											));
										};
										eval_with_context_and_xy_and_z(
											&inlined_node, stack, context, &reserved_vars, cc_x, cc_y, v.0,
											v.1,
										)
									} else if stack.num_args() == 2 {
										eval_with_context_and_xy_and_z(
											&inlined_node,
											stack,
											context,
											&reserved_vars,
											cc_x,
											cc_y,
											stack.get_arg(0).unwrap().as_float()?,
											stack.get_arg(1).unwrap().as_float()?,
										)
									} else {
										return Err(EvalexprError::wrong_function_argument_amount(
											stack.num_args(),
											2,
										));
									}
								} else {
									if stack.num_args() == 1 {
										let Ok(v) = stack.get_arg(0).unwrap().as_float2() else {
											return Err(EvalexprError::wrong_function_argument_amount(
												stack.num_args(),
												2,
											));
										};
										eval_with_context_and_xy(
											&inlined_node, stack, context, &reserved_vars, v.0, v.1,
										)
									} else if stack.num_args() == 2 {
										eval_with_context_and_xy(
											&inlined_node,
											stack,
											context,
											&reserved_vars,
											stack.get_arg(0).unwrap().as_float()?,
											stack.get_arg(1).unwrap().as_float()?,
										)
									} else {
										return Err(EvalexprError::wrong_function_argument_amount(
											stack.num_args(),
											2,
										));
									}
								}
							},
						};

						// tl_context.stack_overflow_guard.set(stack_depth);
						res
					});

					ctx.set_function(istr(&entry.name.trim()), fun);
				}
			}
		},
		EntryType::Integral { func, .. } => {
			if let Err(errs) = func.inline_pass(ctx) {
				return Err((entry.id, errs));
			}
		},
		_ => {},
	}
	Ok(())
}
struct DiscontinuityDetector {
	prev_value:      Option<(f64, f64)>,
	prev_abs_change: f64,
	eps:             f64,
}
impl DiscontinuityDetector {
	fn new(step_size: f64, eps: f64) -> Self {
		Self { prev_value: None, prev_abs_change: 0.0, eps: (step_size * 0.01).max(eps) }
	}
	fn detect(
		&mut self, arg: f64, value: f64, mut eval: impl FnMut(f64) -> Option<f64>,
	) -> Option<((f64, f64), (f64, f64))> {
		let prev_value = self.prev_value.replace((arg, value));
		if value.is_nan() {
			return None;
		}

		if let Some((prev_arg, prev_value)) = prev_value {
			if prev_value.is_nan() {
				return None;
			}
			let abs_change = (value - prev_value).abs();

			// println!("abs_change: {abs_change}");
			if abs_change > self.prev_abs_change * 2.0 {
				let midpoint = (prev_arg + arg) * 0.5;
				let value_at_midpoint = eval(midpoint)?;

				let left_change = (value_at_midpoint - prev_value).abs();
				let right_change = (value - value_at_midpoint).abs();

				let max_half = left_change.max(right_change);
				if max_half > 0.8 * abs_change {
					// let min_half = left_change.min(right_change);
					// let ratio = min_half / max_half;
					// if ratio < 0.2 {
					// println!("max_half {max_half} abs_change {abs_change}");
					// discontinuity detected - find exact points via bisection
					return Some(self.bisect_discontinuity(prev_arg, prev_value, arg, value, &mut eval));
				}
			}

			self.prev_abs_change = abs_change;
		}

		None
	}

	fn bisect_discontinuity(
		&self, mut left_arg: f64, mut left_value: f64, mut right_arg: f64, mut right_value: f64,
		eval: &mut impl FnMut(f64) -> Option<f64>,
	) -> ((f64, f64), (f64, f64)) {
		// Bisect until we're within eps
		while (right_arg - left_arg) > self.eps {
			let mid_arg = (left_arg + right_arg) * 0.5;
			let Some(mid_value) = eval(mid_arg) else {
				break;
			};

			let left_change = (mid_value - left_value).abs();
			let right_change = (right_value - mid_value).abs();

			// Move toward the side with the larger jump
			if left_change < right_change {
				left_arg = mid_arg;
				left_value = mid_value;
			} else {
				right_arg = mid_arg;
				right_value = mid_value;
			}
		}

		((left_arg, left_value), (right_arg, right_value))
	}
}

pub struct PlotParams {
	pub eps:         f64,
	pub first_x:     f64,
	pub last_x:      f64,
	pub first_y:     f64,
	pub last_y:      f64,
	pub step_size:   f64,
	pub step_size_y: f64,
	pub resolution:  usize,
}
#[allow(clippy::too_many_arguments)]
pub fn create_entry_plot_elements<T: EvalexprFloat>(
	entry: &mut Entry<T>, id: Id, sorting_idx: u32, selected_id: Option<Id>,
	ctx: &evalexpr::HashMapContext<T>, plot_params: &PlotParams,
	draw_buffer: &ThreadLocal<crate::DrawBufferRC>, tl_context: &Arc<ThreadLocal<ThreadLocalContext<T>>>,
	reserved_vars: &ReservedVars,
) -> Result<(), Vec<(u64, String)>> {
	// thread_local_get(tl_context).stack_overflow_guard.set(0);

	let visible = entry.visible;
	if !visible && !matches!(entry.ty, EntryType::Folder { .. } | EntryType::Integral { .. }) {
		return Ok(());
	}
	let color = entry.color();
	// println!("step_size: {deriv_step_x}");

	let draw_buffer_c = thread_local_get(draw_buffer);
	match &mut entry.ty {
		EntryType::Folder { entries } => {
			let mut errors = std::sync::Mutex::new(Vec::new());
			entries.par_iter_mut().enumerate().for_each(|(ei, entry)| {
				let eid = Id::new(entry.id);
				if let Err(e) = create_entry_plot_elements(
					entry,
					eid,
					sorting_idx + ei as u32,
					selected_id,
					ctx,
					plot_params,
					draw_buffer,
					tl_context,
					reserved_vars,
				) {
					errors.lock().unwrap().extend(e);
				}
			});
			if errors.get_mut().unwrap().is_empty() {
				return Ok(());
			}
			return Err(errors.into_inner().unwrap());
		},
		EntryType::Constant { .. } => {},
		EntryType::Integral { func, lower, upper, calculated, resolution, style, .. } => {
			let mut stack = thread_local_get(tl_context).stack.borrow_mut();
			let mut draw_buffer = draw_buffer_c.inner.borrow_mut();
			let (Some(lower_node), Some(upper_node), Some(func_node)) =
				(&lower.node, &upper.node, &func.inlined_node)
			else {
				return Ok(());
			};
			let lower = match lower_node.eval_float_with_context(&mut stack, ctx) {
				Ok(lower) => lower.to_f64(),
				Err(e) => {
					return Err(vec![(entry.id, format!("Error evaluating lower bound: {e}"))]);
				},
			};
			let upper = match upper_node.eval_float_with_context(&mut stack, ctx) {
				Ok(upper) => upper.to_f64(),
				Err(e) => {
					return Err(vec![(entry.id, format!("Error evaluating upper bound: {e}"))]);
				},
			};
			let range = upper - lower;
			if lower > upper {
				*calculated = None;

				return Err(vec![(entry.id, "Lower bound must be less than upper bound".to_string())]);
			}
			let resolution = *resolution;
			let step = range / resolution as f64;
			let step_f = f64_to_float::<T>(step);
			if lower + step == lower {
				*calculated = Some(T::ZERO);
				return Ok(());
			}
			*calculated = None;

			// let mut polygons = Vec::with_capacity(resolution);
			let mut int_lines = Vec::with_capacity(resolution);
			let mut fun_lines = Vec::with_capacity(resolution);

			let stroke_color = color;
			// let rgba_color = stroke_color.to_srgba_unmultiplied();
			// let fill_color = Color32::from_rgba_unmultiplied(rgba_color[0], rgba_color[1], rgba_color[1],
			// 128);

			let mut result: T = T::ZERO;
			let mut prev_y: Option<T> = None;
			// let mut prev_sampling_point: Option<(f64, f64)> = None;
			for i in 0..(resolution + 1) {
				let sampling_x = lower + step * i as f64;

				let cur_x = sampling_x;
				let cur_y = match eval_float_with_context_and_x(
					func_node,
					&mut stack,
					ctx,
					reserved_vars,
					f64_to_float::<T>(sampling_x),
				) {
					Ok(y) => {
						y.to_f64()
						// if let Some((prev_x, prev_y)) = prev_sampling_point {
						// 	(cur_x, cur_y) = zoom_in_on_nan_boundary(
						// 		(prev_x, prev_y),
						// 		(sampling_x, y.to_f64()),
						// 		plot_params.eps,
						// 		|x| {
						// 			func.eval_float_with_context_and_x(ctx, &f64_to_value(x))
						// 				.map(|y| y.to_f64())
						// 				.ok()
						// 		},
						// 	)
						// } else {
						// 	cur_x = sampling_x;
						// 	cur_y = y.to_f64();
						// }
						// prev_sampling_point = Some((sampling_x, y.to_f64()));
					},
					Err(e) => {
						return Err(vec![(entry.id, e.to_string())]);
					},
				};

				let y = f64_to_float::<T>(cur_y);
				if let Some(prev_y) = prev_y {
					// let eps = 0.0;
					let prev_y_f64 = prev_y.to_f64();

					if prev_y_f64.signum() != cur_y.signum() {
						//2 triangles
						let diff = (prev_y_f64 - cur_y).abs();
						let t = prev_y_f64.abs() / diff;

						let t = f64_to_float::<T>(t);

						let step1 = step_f * t;
						let step2 = step_f - step1;

						let b1 = prev_y * step1;
						let b2 = y * step2;
						result = result + b1 * T::HALF;
						result = result + b2 * T::HALF;
					} else {
						let dy = y - prev_y;
						let step = f64_to_float::<T>(step);
						let d = dy * step;
						result = result + prev_y * step;
						result = result + d * T::HALF;
					}
				}
				if result.is_nan() {
					return Err(vec![(entry.id, "Integral is undefined".to_string())]);
				}

				if visible {
					if style.show_integral_fn {
						int_lines.push(PlotPoint::new(cur_x, result.to_f64()));
					}
					if style.show_function {
						fun_lines.push(PlotPoint::new(cur_x, cur_y));
					}
				}
				prev_y = Some(y);
				// x += step;
			}
			// draw_buffer.polygons.push(DrawPolygonGroup::new(sorting_idx, polygons));
			let int_name = if entry.name.is_empty() { func.text.as_str() } else { entry.name.as_str() };
			draw_buffer.lines.push(DrawLine::new(
				sorting_idx,
				Id::NULL,
				1.0,
				Line::new(format!("∫[{},x]({})dx", lower, int_name.trim()), PlotPoints::Owned(int_lines))
					.color(stroke_color),
			));

			let mut fn_line = Line::new("", PlotPoints::Owned(fun_lines)).color(stroke_color);
			if style.show_area {
				fn_line = fn_line.fill(0.0).fill_alpha(0.3);
			}
			draw_buffer.lines.push(DrawLine::new(sorting_idx, id, 1.0, fn_line));
			*calculated = Some(result);
		},
		EntryType::Label { x, y, size, underline, .. } => {
			let mut draw_buffer = draw_buffer_c.inner.borrow_mut();
			let mut stack = thread_local_get(tl_context).stack.borrow_mut();

			match eval_point(&mut stack, ctx, reserved_vars, x.node.as_ref(), y.node.as_ref()) {
				Ok(Some((x, y))) => {
					let size = if let Some(size) = &size.node {
						match eval_float_with_context_and_x(
							size,
							&mut stack,
							ctx,
							reserved_vars,
							f64_to_float::<T>(x),
						) {
							Ok(size) => size.to_f64() as f32,
							Err(e) => {
								return Err(vec![(entry.id, e.to_string())]);
							},
						}
					} else {
						12.0
					};
					let mut label_text = RichText::new(entry.name.clone()).size(size);
					if *underline {
						label_text = label_text.underline()
					}

					let text = Text::new(entry.name.clone(), PlotPoint { x, y }, label_text).color(color);

					draw_buffer.texts.push(DrawText::new(sorting_idx, text));
				},
				Err(e) => {
					return Err(vec![(entry.id, e)]);
				},
				_ => {},
			}
		},
		EntryType::Points { points, style } => {
			let mut stack = thread_local_get(tl_context).stack.borrow_mut();
			let mut draw_buffer = draw_buffer_c.inner.borrow_mut();
			// main_context
			// 	.write()
			// 	.unwrap()
			// 	.set_value("x", evalexpr::Value::<T>::Float(T::ZERO))
			// 	.unwrap();
			let mut arrow_buffer = vec![];
			let mut line_buffer = vec![];
			let color_rgba = color.to_array();
			let color_outer =
				Color32::from_rgba_unmultiplied(color_rgba[0], color_rgba[1], color_rgba[1], 128);
			let mut prev_point: Option<egui::Vec2> = None;
			let arrow_scale = egui::Vec2::new(
				(plot_params.last_x - plot_params.first_x) as f32,
				(plot_params.last_y - plot_params.first_y) as f32,
			) * 0.002;
			let points_len = points.len();
			for (i, p) in points.iter_mut().enumerate() {
				match eval_point(&mut stack, ctx, reserved_vars, p.x.node.as_ref(), p.y.node.as_ref()) {
					Ok(Some((x, y))) => {
						p.val = Some((f64_to_float::<T>(x), f64_to_float::<T>(y)));
						let point_id = id.with(i);
						let selected = selected_id == Some(id);
						let radius = if selected { 6.5 } else { 4.5 };
						let radius_outer = if selected { 12.5 } else { 7.5 };

						if style.show_points || p.drag_point.is_some() {
							draw_buffer.points.push(DrawPoint::new(
								sorting_idx,
								i as u32,
								PointInteraction {
									x,
									y,
									radius,
									ty: PointInteractionType::Other(OtherPointType::Point),
								},
								Points::new(entry.name.clone(), [x, y]).color(color).radius(radius),
							));
							if p.drag_point.is_some() {
								let selectable_point = PointInteraction {
									ty: PointInteractionType::Draggable { i: (id, i as u32) },
									x,
									y,
									radius: radius_outer,
								};
								draw_buffer.points.push(DrawPoint::new(
									sorting_idx,
									i as u32,
									selectable_point,
									Points::new(entry.name.clone(), [x, y])
										.id(point_id)
										.color(color_outer)
										.radius(radius_outer),
								));
							}
						}
						if let Some(label_config) = &style.label_config
							&& i == points_len - 1
							&& !entry.name.trim().is_empty()
						{
							let size = label_config.size.size();
							let mut label = RichText::new(entry.name.clone()).size(size);
							if label_config.italic {
								label = label.italics();
							}

							let dir = label_config.pos.dir();
							let text = Text::new(
								entry.name.clone(),
								PlotPoint {
									x: x + (dir.x * size * arrow_scale.x) as f64,
									y: y + (dir.y * size * arrow_scale.y) as f64,
								},
								label,
							)
							.color(color);

							draw_buffer.texts.push(DrawText::new(sorting_idx, text));
						}

						if style.show_lines {
							line_buffer.push([x, y]);
							let cur_point = egui::Vec2::new(x as f32, y as f32);
							if style.show_arrows
								&& let Some(pp) = prev_point
							{
								let dir = (pp - cur_point).normalized();
								let arrow_len = arrow_scale * radius_outer;
								let base = cur_point + dir * arrow_len.length();
								let a = base + dir.rot90() * arrow_len * 0.5;
								let b = base - dir.rot90() * arrow_len * 0.5;

								arrow_buffer.push(
									Polygon::new(
										"",
										vec![[x, y], [a.x as f64, a.y as f64], [b.x as f64, b.y as f64]],
									)
									.fill_color(color)
									.allow_hover(false)
									.stroke(Stroke::new(0.0, color)),
								);
							}
							prev_point = Some(cur_point);
						}
					},
					Err(e) => {
						return Err(vec![(entry.id, e)]);
					},
					_ => {},
				}
			}

			// let width = if selected { 3.5 } else { 1.0 };
			let width = 1.0;

			if line_buffer.len() > 1 && style.show_lines {
				let line = Line::new(entry.name.clone(), line_buffer)
					.color(color)
					.id(id)
					.width(style.line_style.line_width)
					.style(style.line_style.egui_line_style());
				draw_buffer.lines.push(DrawLine::new(sorting_idx, id, width, line));
				if !arrow_buffer.is_empty() {
					draw_buffer.polygons.push(DrawPolygonGroup::new(sorting_idx, arrow_buffer));
				}
				// plot_ui.line(line);
			}
		},
		EntryType::Function {
			func,
			can_be_drawn,
			range_start,
			range_end,
			style,
			ty,
			implicit_resolution,
			..
		} => {
			if !*can_be_drawn {
				return Ok(());
			}
			let name = if entry.name.is_empty() {
				match ty {
					FunctionType::Y => format!("f(y): {}", func.text.trim()),
					FunctionType::Ranged | FunctionType::X => {
						format!("f(x):  {}", func.text.trim())
					},
					FunctionType::Implicit { complex } => {
						if *complex {
							format!("F(zx,zy): {}", func.text.trim())
						} else {
							func.text.clone()
						}
					},
				}
			} else {
				match ty {
					FunctionType::Y => format!("{}(y):  {}", entry.name.trim(), func.text.trim()),
					FunctionType::Ranged | FunctionType::X => {
						format!("{}(x): {}", entry.name.trim(), func.text.trim())
					},
					FunctionType::Implicit { complex } => {
						if *complex {
							format!("{}(zx,zy): {}", entry.name.trim(), func.text.trim())
						} else {
							format!("{}: {}", entry.name.trim(), func.text.trim())
						}
					},
				}
			};
			let selected = selected_id == Some(id);
			let width = if selected { style.line_width + 2.5 } else { style.line_width };

			// let mut cache = (!animating).then(|| {
			// 	state
			// 		.points_cache
			// 		.entry(text.clone())
			// 		.or_insert_with(|| AHashMap::with_capacity(ui_state.conf.resolution))
			// });
			let add_line = |line: Vec<PlotPoint>| {
				let mut draw_buffer = draw_buffer_c.inner.borrow_mut();
				draw_buffer.lines.push(DrawLine::new(
					sorting_idx,
					id,
					width,
					Line::new(&name, PlotPoints::Owned(line))
						.id(id)
						.width(width)
						.style(style.egui_line_style())
						.color(color),
				));
			};

			if let Some(expr_func) = &func.expr_function {
				match ty {
					FunctionType::X => {
						let mut stack = thread_local_get(tl_context).stack.borrow_mut();
						let mut pp_buffer = vec![];
						let mut prev_sampling_point: Option<(f64, f64)> = None;
						let mut sampling_x = plot_params.first_x;
						if let Some(constant) = expr_func.as_constant() {
							let value = constant.as_float().map_err(|e| vec![(entry.id, e.to_string())])?;
							pp_buffer.push(PlotPoint::new(plot_params.first_x, value.to_f64()));
							pp_buffer.push(PlotPoint::new(plot_params.last_x, value.to_f64()));
						} else {
							// let mut prev_y = None;
							let mut prev_first_derivative = 0.0;
							let mut discontinuity_detector =
								DiscontinuityDetector::new(plot_params.step_size, plot_params.eps);

							while sampling_x < plot_params.last_x {
								match expr_func.call(&mut stack, ctx, &[f64_to_value::<T>(sampling_x)]) {
									// eval_with_context_and_x(
									// func_node,
									// &mut stack,
									// ctx,
									// reserved_vars,
									// f64_to_float::<T>(sampling_x),)
									Ok(Value::Float(y)) => {
										let y = y.to_f64();

										let zoomed_in_on_nan_boundary =
											prev_sampling_point.and_then(|(prev_x, prev_y)| {
												zoom_in_x_on_nan_boundary(
													(prev_x, prev_y),
													(sampling_x, y),
													plot_params.eps,
													|x| {
														expr_func
															.call(&mut stack, ctx, &[f64_to_value::<T>(x)])
															// eval_float_with_context_and_x(
															// 	func_node,
															// 	&mut stack,
															// 	ctx,
															// 	reserved_vars,
															// 	f64_to_float::<T>(x),
															// )
															.and_then(|y| y.as_float())
															.map(|y| y.to_f64())
															.ok()
													},
												)
											});

										let mut on_nan_boundary = false;
										let (cur_x, cur_y) = if let Some(zoomed_in_on_nan_boundary) =
											zoomed_in_on_nan_boundary
										{
											on_nan_boundary = true;
											zoomed_in_on_nan_boundary
										} else {
											(sampling_x, y)
										};

										if !on_nan_boundary
											&& let Some((left, right)) =
												discontinuity_detector.detect(cur_x, cur_y, |x| {
													expr_func
														.call(&mut stack, ctx, &[f64_to_value::<T>(x)])
														.and_then(|y| y.as_float())
														.map(|y| y.to_f64())
														.ok()
												}) {
											pp_buffer.push(PlotPoint::new(left.0, left.1));
											// start a new line
											add_line(mem::take(&mut pp_buffer));
											pp_buffer.push(PlotPoint::new(right.0, right.1));
										} else {
											if !cur_y.is_nan() {
												pp_buffer.push(PlotPoint::new(cur_x, cur_y));
											}
										}

										prev_sampling_point = Some((sampling_x, y));
									},
									Ok(Value::Empty) => {},
									Ok(_) => {
										return Err(vec![(
											entry.id,
											"Function must return float or empty".to_string(),
										)]);
									},

									Err(e) => {
										return Err(vec![(entry.id, e.to_string())]);
									},
								}

								let prev_x = sampling_x;
								sampling_x += plot_params.step_size;
								if sampling_x == prev_x {
									break;
								}
							}
						}

						add_line(pp_buffer);
					},
					FunctionType::Y => {
						todo!()
					},
					_ => {
						todo!()
					},
				}
			} else if let Some(func_node) = &func.inlined_node {
				match ty {
					FunctionType::X => {},

					FunctionType::Y => {
						let mut stack = thread_local_get(tl_context).stack.borrow_mut();
						let mut sampling_y = plot_params.first_y;
						let mut pp_buffer = vec![];
						let mut prev_sampling_point: Option<(f64, f64)> = None;
						if let Some(constant) = func_node.as_constant() {
							let value = constant.as_float().map_err(|e| vec![(entry.id, e.to_string())])?;
							pp_buffer.push(PlotPoint::new(plot_params.first_y, value.to_f64()));
							pp_buffer.push(PlotPoint::new(plot_params.last_y, value.to_f64()));
						} else {
							let mut discontinuity_detector =
								DiscontinuityDetector::new(plot_params.step_size_y, plot_params.eps);
							while sampling_y < plot_params.last_y {
								match eval_with_context_and_y(
									func_node,
									&mut stack,
									ctx,
									reserved_vars,
									f64_to_float::<T>(sampling_y),
								) {
									Ok(Value::Float(x)) => {
										let x = x.to_f64();

										let zoomed_in_on_nan_boundary =
											prev_sampling_point.and_then(|(prev_x, prev_y)| {
												zoom_in_y_on_nan_boundary(
													(prev_x, prev_y),
													(x, sampling_y),
													plot_params.eps,
													|y| {
														eval_float_with_context_and_y(
															func_node,
															&mut stack,
															ctx,
															reserved_vars,
															f64_to_float::<T>(y),
														)
														.map(|x| x.to_f64())
														.ok()
													},
												)
											});
										let mut on_nan_boundary = false;
										let (cur_x, cur_y) = if let Some(zoomed_in_on_nan_boundary) =
											zoomed_in_on_nan_boundary
										{
											on_nan_boundary = true;
											zoomed_in_on_nan_boundary
										} else {
											(x, sampling_y)
										};
										prev_sampling_point = Some((x, sampling_y));

										if !on_nan_boundary
											&& let Some((left, right)) =
												discontinuity_detector.detect(cur_y, cur_x, |y| {
													eval_float_with_context_and_y(
														func_node,
														&mut stack,
														ctx,
														reserved_vars,
														f64_to_float::<T>(y),
													)
													.map(|x| x.to_f64())
													.ok()
												}) {
											pp_buffer.push(PlotPoint::new(left.1, left.0));
											// start a new line
											add_line(mem::take(&mut pp_buffer));
											pp_buffer.push(PlotPoint::new(right.1, right.0));
										} else {
											if !cur_x.is_nan() {
												pp_buffer.push(PlotPoint::new(cur_x, cur_y));
											}
										}
									},
									Ok(Value::Empty) => {},
									Ok(_) => {
										return Err(vec![(
											entry.id,
											"Function must return float or empty".to_string(),
										)]);
									},

									Err(e) => {
										return Err(vec![(entry.id, e.to_string())]);
									},
								}

								let prev_y = sampling_y;
								sampling_y += plot_params.step_size_y;
								if sampling_y == prev_y {
									break;
								}
							}
						}

						add_line(pp_buffer);
					},
					FunctionType::Ranged => {
						let mut stack = thread_local_get(tl_context).stack.borrow_mut();
						let mut pp_buffer = vec![];
						match eval_point(
							&mut stack,
							ctx,
							reserved_vars,
							range_start.node.as_ref(),
							range_end.node.as_ref(),
						) {
							Ok(Some((start, end))) => {
								if start > end {
									return Err(vec![(
										entry.id,
										"Range start must be less than range end".to_string(),
									)]);
								}
								let range = end - start;
								let step = range / plot_params.resolution as f64;
								if start + step == end {
									return Ok(());
								}
								let mut discontinuity_detector =
									DiscontinuityDetector::new(step, plot_params.eps);
								for i in 0..(plot_params.resolution + 1) {
									let x = start + step * i as f64;
									match eval_with_context_and_x(
										func_node,
										&mut stack,
										ctx,
										reserved_vars,
										f64_to_float::<T>(x),
									) {
										Ok(Value::Float(y)) => {
											if let Some((left, right)) =
												discontinuity_detector.detect(x, y.to_f64(), |x| {
													eval_float_with_context_and_x(
														func_node,
														&mut stack,
														ctx,
														reserved_vars,
														f64_to_float::<T>(x),
													)
													.map(|y| y.to_f64())
													.ok()
												}) {
												pp_buffer.push(PlotPoint::new(left.0, left.0));
												add_line(mem::take(&mut pp_buffer));
												pp_buffer.push(PlotPoint::new(right.0, right.0));
											} else {
												if !y.is_nan() {
													pp_buffer.push(PlotPoint::new(x, y.to_f64()));
												}
											}
										},
										Ok(Value::Empty) => {},
										Ok(Value::Float2(x, y)) => {
											// TODO: detect discontinuity for ranged thar teturn Float2

											// if let Some((left, right)) =
											// 	discontinuity_detector.detect(x.to_f64(), y.to_f64(), |x| {
											// 		eval_with_context_and_x(
											// 			func_node,
											// 			&mut stack,
											// 			ctx,
											// 			reserved_vars,
											// 			f64_to_float::<T>(x),
											// 		)
											// 		.and_then(|y| y.as_float2())
											// 		.map(|y| y.1.to_f64())
											// 		.ok()
											// 	}) {
											// 	pp_buffer.push(PlotPoint::new(left.0, left.0));
											// 	add_line(mem::take(&mut pp_buffer));
											// 	pp_buffer.push(PlotPoint::new(right.0, right.0));
											// } else {
											// 	if !y.is_nan() {
											// 		pp_buffer.push(PlotPoint::new(x.to_f64(), y.to_f64()));
											// 	}
											// }
											if y.is_nan() {
												if !pp_buffer.is_empty() {
													add_line(mem::take(&mut pp_buffer));
												}
											} else {
												pp_buffer.push(PlotPoint::new(x.to_f64(), y.to_f64()));
											}
										},
										Ok(_) => {
											return Err(vec![(
												entry.id,
												"Ranged function must return 1 or 2 float values".to_string(),
											)]);
										},
										Err(e) => {
											return Err(vec![(entry.id, e.to_string())]);
										},
									}
								}
								add_line(pp_buffer);
							},
							Err(e) => {
								return Err(vec![(entry.id, e)]);
							},
							_ => {},
						}
					},
					FunctionType::Implicit { .. } => {
						let mins = (plot_params.first_x, plot_params.first_y);
						let maxs = (plot_params.last_x, plot_params.last_y);

						struct ImplicitContext<'a, T: EvalexprFloat> {
							stack: RefMut<'a, Stack<T>>,
							x:     &'a Cell<T>,
							y:     &'a Cell<T>,
						}
						for (_, lines) in marching_squares::marching_squares(
							|cc: &mut ImplicitContext<T>, x, y| {
								let xf = f64_to_float::<T>(x);
								let yf = f64_to_float::<T>(y);

								cc.x.set(xf);
								cc.y.set(yf);
								eval_float_with_context_and_xy_and_z(
									func_node,
									&mut cc.stack,
									ctx,
									reserved_vars,
									f64_to_float::<T>(x),
									f64_to_float::<T>(y),
									f64_to_float::<T>(x),
									f64_to_float::<T>(y),
								)
								.map(|y| y.to_f64())
								.map_err(|e| e.to_string())
							},
							mins,
							maxs,
							*implicit_resolution,
							|| {
								let tl_context = thread_local_get(tl_context);
								// tl_context.stack_overflow_guard.set(0);

								ImplicitContext {
									stack: tl_context.stack.borrow_mut(),
									x:     &tl_context.cc_x,
									y:     &tl_context.cc_y,
								}
							},
							&thread_local_get(tl_context).marching_squares_cache,
						)
						.map_err(|e| vec![(entry.id, e)])?
						{
							for line in lines {
								add_line(line);
							}
						}
					},
				}
			}
		},
	}
	Ok(())
}

pub fn eval_point<T: EvalexprFloat>(
	stack: &mut Stack<T>, ctx: &evalexpr::HashMapContext<T>, reserved_vars: &ReservedVars,
	px: Option<&FlatNode<T>>, py: Option<&FlatNode<T>>,
) -> Result<Option<(f64, f64)>, String> {
	let (Some(x), Some(y)) = (px, py) else {
		return Ok(None);
	};
	let x = eval_float_with_context_and_x(x, stack, ctx, reserved_vars, T::ZERO).map_err(|e| e.to_string())?;
	let y = eval_float_with_context_and_x(y, stack, ctx, reserved_vars, T::ZERO).map_err(|e| e.to_string())?;
	Ok(Some((x.to_f64(), y.to_f64())))
}
fn zoom_in_x_on_nan_boundary(
	a: (f64, f64), b: (f64, f64), eps: f64, mut eval: impl FnMut(f64) -> Option<f64>,
) -> Option<(f64, f64)> {
	// If both are nans or both are defined, no need to do anything
	if a.1.is_nan() == b.1.is_nan() {
		return None;
	}

	let mut left = a;
	let mut right = b;

	let mut prev_mid_x = None;
	while (right.0 - left.0).abs() > eps {
		let mid_x = (left.0 + right.0) * 0.5;
		let mid_y = eval(mid_x)?;
		if prev_mid_x == Some(mid_x) {
			break;
		}
		prev_mid_x = Some(mid_x);

		let mid = (mid_x, mid_y);

		if left.1.is_nan() == mid_y.is_nan() {
			left = mid;
		} else {
			right = mid;
		}
	}

	Some(if left.1.is_nan() { right } else { left })
}
fn zoom_in_y_on_nan_boundary(
	a: (f64, f64), b: (f64, f64), eps: f64, mut eval: impl FnMut(f64) -> Option<f64>,
) -> Option<(f64, f64)> {
	// If both are nans or both are defined, no need to do anything
	if a.0.is_nan() == b.0.is_nan() {
		return None;
	}

	let mut left = a;
	let mut right = b;

	let mut prev_mid_y = None;
	while (right.1 - left.1).abs() > eps {
		let mid_y = (left.1 + right.1) * 0.5;
		let mid_x = eval(mid_y)?;
		if prev_mid_y == Some(mid_y) {
			break;
		}
		prev_mid_y = Some(mid_y);

		let mid = (mid_x, mid_y);

		if left.0.is_nan() == mid_x.is_nan() {
			left = mid;
		} else {
			right = mid;
		}
	}

	Some(if left.0.is_nan() { right } else { left })
}

pub fn f64_to_value<T: EvalexprFloat>(x: f64) -> Value<T> { Value::<T>::Float(T::f64_to_float(x)) }
pub fn f64_to_float<T: EvalexprFloat>(x: f64) -> T { T::f64_to_float(x) }

#[derive(Clone, Debug)]
pub enum DragPoint {
	BothCoordLiterals,
	XLiteral,
	YLiteral,
	XConstant(IStr),
	YConstant(IStr),
	XLiteralYConstant(IStr),
	YLiteralXConstant(IStr),
	BothCoordConstants(IStr, IStr),
	SameConstantBothCoords(IStr),
}

pub struct NodeAnalysis<'a> {
	pub is_literal:      bool,
	pub num_identifiers: u32,
	pub constants:       SmallVec<[&'a str; 6]>,
}
pub fn analyze_node<T: EvalexprFloat>(node: &FlatNode<T>) -> NodeAnalysis<'_> {
	let mut is_literal = true;
	let mut constants = SmallVec::new();
	let mut num_identifiers = 0;

	for i in node.iter_variable_identifiers() {
		constants.push(i);
		is_literal = false;
	}

	for _ in node.iter_identifiers() {
		is_literal = false;
		num_identifiers += 1;
	}

	NodeAnalysis { is_literal, constants, num_identifiers }
}
pub fn preprecess_fn(text: &str) -> Result<Option<String>, String> {
	// regex to check if theres an y identifier (single y not surrounded by alphanumerics on either
	// side)
	static RE_Y: LazyLock<Regex> =
		LazyLock::new(|| Regex::new(r"(?:^|[^a-zA-Z0-9])y(?:$|[^a-zA-Z0-9])").unwrap());
	static RE_X: LazyLock<Regex> =
		LazyLock::new(|| Regex::new(r"(?:^|[^a-zA-Z0-9])x(?:$|[^a-zA-Z0-9])").unwrap());
	static RE_ZY: LazyLock<Regex> =
		LazyLock::new(|| Regex::new(r"(?:^|[^a-zA-Z0-9])zy(?:$|[^a-zA-Z0-9])").unwrap());
	static RE_ZX: LazyLock<Regex> =
		LazyLock::new(|| Regex::new(r"(?:^|[^a-zA-Z0-9])zx(?:$|[^a-zA-Z0-9])").unwrap());

	let text_b = text.as_bytes();
	let mut split = None;
	for (i, &c) in text_b.iter().enumerate() {
		if c == b'=' {
			let prev = text_b.get(i - 1).copied();
			let next = text_b.get(i + 1).copied();
			static IGNORE_PREV: &[u8] = b"!=<>+-*/%^|&";
			if next != Some(b'=') && IGNORE_PREV.iter().all(|&c| Some(c) != prev) {
				split = Some((&text[0..i], text.get(i + 1..).unwrap_or("")));
				break;
			}
		}
	}

	let Some((left, right)) = split else {
		return Ok(None);
	};
	let left = left.trim();
	let right = right.trim();
	if left.is_empty() || right.is_empty() {
		return Err("Something is needed on both sides of the = sign".to_string());
	}
	if left == "y" {
		if RE_Y.is_match(right) || RE_ZY.is_match(right) || RE_ZX.is_match(right) {
			return Ok(Some(format!("y - ({right})")));
		}
		return Ok(Some(right.to_string()));
	}
	if left == "x" {
		if RE_X.is_match(right) || RE_ZY.is_match(right) || RE_ZX.is_match(right) {
			return Ok(Some(format!("x - ({right})")));
		}
		if RE_Y.is_match(right) {
			return Ok(Some(right.to_string()));
		}
		return Ok(Some(format!("{right} + y*0")));
	}
	let new = format!("{} - ({})", left, right);
	Ok(Some(new))
}

fn eval_with_context_and_x<T: EvalexprFloat>(
	node: &FlatNode<T>, stack: &mut Stack<T>, context: &evalexpr::HashMapContext<T>,
	reserved_vars: &ReservedVars, x: T,
) -> EvalexprResultValue<T> {
	node.eval_with_context_and_override(stack, context, &[(reserved_vars.x, x)])
}
fn eval_with_context_and_y<T: EvalexprFloat>(
	node: &FlatNode<T>, stack: &mut Stack<T>, context: &evalexpr::HashMapContext<T>,
	reserved_vars: &ReservedVars, y: T,
) -> EvalexprResultValue<T> {
	node.eval_with_context_and_override(stack, context, &[(reserved_vars.y, y)])
}
fn eval_float_with_context_and_x<T: EvalexprFloat>(
	node: &FlatNode<T>, stack: &mut Stack<T>, context: &evalexpr::HashMapContext<T>,
	reserved_vars: &ReservedVars, x: T,
) -> EvalexprResult<T, T> {
	node.eval_float_with_context_and_override(stack, context, &[(reserved_vars.x, x)])
}
fn eval_float_with_context_and_y<T: EvalexprFloat>(
	node: &FlatNode<T>, stack: &mut Stack<T>, context: &evalexpr::HashMapContext<T>,
	reserved_vars: &ReservedVars, y: T,
) -> EvalexprResult<T, T> {
	node.eval_float_with_context_and_override(stack, context, &[(reserved_vars.y, y)])
}

fn eval_with_context_and_xy_and_z<T: EvalexprFloat>(
	node: &FlatNode<T>, stack: &mut Stack<T>, context: &evalexpr::HashMapContext<T>,
	reserved_vars: &ReservedVars, x: T, y: T, zx: T, zy: T,
) -> EvalexprResultValue<T> {
	node.eval_with_context_and_override(
		stack,
		context,
		&[(reserved_vars.x, x), (reserved_vars.y, y), (reserved_vars.zx, zx), (reserved_vars.zy, zy)],
	)
}
fn eval_float_with_context_and_xy_and_z<T: EvalexprFloat>(
	node: &FlatNode<T>, stack: &mut Stack<T>, context: &evalexpr::HashMapContext<T>,
	reserved_vars: &ReservedVars, x: T, y: T, zx: T, zy: T,
) -> EvalexprResult<T, T> {
	match eval_with_context_and_xy_and_z(node, stack, context, reserved_vars, x, y, zx, zy) {
		Ok(Value::Float(float)) => Ok(float),
		Ok(value) => Err(EvalexprError::expected_float(value)),
		Err(error) => Err(error),
	}
}
fn eval_with_context_and_xy<T: EvalexprFloat>(
	node: &FlatNode<T>, stack: &mut Stack<T>, context: &evalexpr::HashMapContext<T>,
	reserved_vars: &ReservedVars, x: T, y: T,
) -> EvalexprResultValue<T> {
	node.eval_with_context_and_override(stack, context, &[(reserved_vars.x, x), (reserved_vars.y, y)])
}
