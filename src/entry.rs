use core::ops::RangeInclusive;

use eframe::egui::{self, Color32};
use evalexpr::{EvalexprFloat, ExpressionFunction, FlatNode, HashMapContext, IStr, Stack, ThinVec, Value, istr_empty};
use serde::{Deserialize, Serialize};

use crate::custom_rendering::fan_fill_renderer::FillRule;
use crate::draw_buffer::DrawBufferScheduler;

mod drag_point;
mod entry_plot_elements;
mod entry_processing;
mod entry_ui;

pub use drag_point::point_dragging;
pub use entry_plot_elements::{PlotParams, schedule_entry_create_plot_elements};
pub use entry_processing::{optimize_entries, prepare_constants, prepare_entries, preprocess_ast};
pub use entry_ui::entry_ui;

pub const DEFAULT_IMPLICIT_RESOLUTION: usize = 200;
pub const MAX_IMPLICIT_RESOLUTION: usize = 500;
pub const MIN_IMPLICIT_RESOLUTION: usize = 10;

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

pub struct Entry<T: EvalexprFloat> {
	pub id:                    u64,
	pub name:                  String,
	pub active:                bool,
	pub color:                 usize,
	pub ty:                    EntryType<T>,
	pub draw_buffer_scheduler: DrawBufferScheduler,
}
pub struct ClonedEntry<T: EvalexprFloat> {
	pub id:    u64,
	pub color: usize,
	pub ty:    EntryType<T>,
}
impl<T: EvalexprFloat> ClonedEntry<T> {
	fn color(&self) -> Color32 { COLORS[self.color % NUM_COLORS] }
}

impl<T: EvalexprFloat + core::fmt::Debug> core::fmt::Debug for Entry<T> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		f.debug_struct("Entry")
			.field("id", &self.id)
			.field("name", &self.name)
			.field("active", &self.active)
			.field("color", &self.color)
			.field("ty", &self.ty)
			.finish()
	}
}

impl<T: EvalexprFloat + Clone> Clone for Entry<T> {
	fn clone(&self) -> Self {
		Self {
			id:                    self.id,
			name:                  self.name.clone(),
			active:                self.active,
			color:                 self.color,
			ty:                    self.ty.clone(),
			draw_buffer_scheduler: DrawBufferScheduler::new(),
		}
	}
}
impl<T: EvalexprFloat> core::hash::Hash for Entry<T> {
	fn hash<H: core::hash::Hasher>(&self, state: &mut H) { self.id.hash(state); }
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum EquationType {
	#[default]
	None,
	Equality,
	LessThan,
	LessThanOrEqual,
	GreaterThan,
	GreaterThanOrEqual,
}

#[derive(Clone, Debug)]
pub struct Expr<T: EvalexprFloat> {
	pub text:             String,
	pub display_rational: bool,
	pub node:             Option<FlatNode<T>>,
	pub expr_function:    Option<ExpressionFunction<T>>,
	pub args:             Vec<IStr>,
	pub equation_type:    EquationType,
}

impl<T: EvalexprFloat> Default for Expr<T> {
	fn default() -> Self {
		Self {
			text:             Default::default(),
			display_rational: false,
			node:             Default::default(),
			args:             Default::default(),
			expr_function:    Default::default(),
			equation_type:    Default::default(),
		}
	}
}
impl<T: EvalexprFloat> Expr<T> {
	fn computed_const(&self) -> Option<Value<T>> {
		let Some(node) = &self.node else { return None };
		if node.as_constant().is_some() {
			// if unoptimized node is constant, no need to display it
			return None;
		}
		if let Some(func) = &self.expr_function { func.as_constant() } else { None }
	}
	fn from_text(text: &str) -> Self {
		// TODO preprocess here too
		Self {
			node:             evalexpr::build_flat_node::<T>(text).ok(),
			equation_type:    EquationType::None,
			display_rational: false,
			expr_function:    None,
			text:             text.to_string(),
			args:             Vec::new(),
		}
	}

	fn args_to_string(&self) -> String {
		let mut args_string = String::new();
		for (i, arg) in self.args.iter().enumerate() {
			if i > 0 {
				args_string.push_str(", ");
			}
			args_string.push_str(arg.to_str());
		}
		args_string
	}
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FunctionType {
	// ExpressionX,
	// ExpressionY,
	Expression,
	WithCustomParams,
}
#[derive(Clone, Debug)]
pub enum EntryType<T: EvalexprFloat> {
	Function {
		can_be_drawn: bool,
		identifier:   IStr,
		func:         Expr<T>,
		style:        LineStyleConfig,
		ty:           FunctionType,

		parametric:          bool,
		parametric_fill:     bool,
		fill_rule:           FillRule,
		/// used when parametric is `true`
		range_start:         Expr<T>,
		/// used when parametric is `true`
		range_end:           Expr<T>,
		/// used for functions with x,y params
		implicit_resolution: usize,
	},
	Constant {
		istr_name: IStr,
		value:     T,
		step:      f64,
		ty:        ConstantType,

		range_start: Expr<T>,
		range_end:   Expr<T>,
	},
	Points {
		identifier: IStr,
		points_ty:     PointsType<T>,
		style:      PointStyle<T>,
	},
	Folder {
		entries: Vec<Entry<T>>,
	},
}

#[derive(Clone, Copy, Default, PartialEq, Debug, Serialize, Deserialize)]
pub enum LabelSize {
	#[default]
	Small,
	Medium,
	Large,
}
impl LabelSize {
	pub fn size(self) -> f32 {
		match self {
			LabelSize::Small => 10.0,
			LabelSize::Medium => 16.0,
			LabelSize::Large => 20.0,
		}
	}
}
#[derive(Clone, Copy, Default, PartialEq, Debug, Serialize, Deserialize)]
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
	Center,
}
impl LabelPosition {
	fn symbol(self) -> &'static str {
		match self {
			LabelPosition::Top => "⮉",
			LabelPosition::Bottom => "⮋",
			LabelPosition::Left => "⮈",
			LabelPosition::Right => "⮊",
			LabelPosition::TLeft => "⬉",
			LabelPosition::TRight => "⬈",
			LabelPosition::BLeft => "⬋",
			LabelPosition::BRight => "⬊",
			LabelPosition::Center => "o",
		}
	}
	fn dir(self) -> egui::Vec2 {
		match self {
			LabelPosition::Top => egui::Vec2::new(0.0, 1.0),
			LabelPosition::Bottom => egui::Vec2::new(0.0, -1.0),
			LabelPosition::Left => egui::Vec2::new(-1.0, 0.0),
			LabelPosition::Right => egui::Vec2::new(1.0, 0.0),
			LabelPosition::TLeft => egui::Vec2::new(-0.71, 0.71),
			LabelPosition::TRight => egui::Vec2::new(0.71, 0.71),
			LabelPosition::BLeft => egui::Vec2::new(-0.71, -0.71),
			LabelPosition::BRight => egui::Vec2::new(0.71, -0.71),
			LabelPosition::Center => egui::Vec2::new(0.0, 0.0),
		}
	}
}

#[derive(Clone, Debug)]
pub struct LabelConfig<T: EvalexprFloat> {
	pub text:   String,
	pub size:   LabelSize,
	pub pos:    LabelPosition,
	pub italic: bool,
	pub angle:  Expr<T>,
}
impl<T: EvalexprFloat> Default for LabelConfig<T> {
	fn default() -> Self {
		Self {
			text:   String::new(),
			size:   LabelSize::Small,
			pos:    LabelPosition::Bottom,
			italic: false,
			angle:  Expr::default(),
		}
	}
}

#[derive(Clone, Debug)]
pub struct PointStyle<T: EvalexprFloat> {
	pub show_lines:             bool,
	pub show_arrows:            bool,
	pub show_points:            bool,
	pub line_style:             LineStyleConfig,
	pub label_config:           Option<LabelConfig<T>>,
	pub fill:                   bool,
	pub fill_rule:              FillRule,
	pub connect_first_and_last: bool,
}

impl<T: EvalexprFloat> Default for PointStyle<T> {
	fn default() -> Self {
		Self {
			show_lines:             true,
			show_points:            true,
			show_arrows:            false,
			label_config:           None,
			line_style:             LineStyleConfig::new(false),
			fill:                   false,
			connect_first_and_last: false,
			fill_rule:              FillRule::default(),
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
pub struct LineStyleConfig {
	#[serde(default)]
	selectable:      bool,
	line_width:      f32,
	#[serde(default)]
	line_style:      LineStyleType,
	line_style_size: f32,
}
impl Default for LineStyleConfig {
	fn default() -> Self {
		Self {
			selectable:      false,
			line_width:      1.5,
			line_style:      LineStyleType::Solid,
			line_style_size: 5.5,
		}
	}
}
impl LineStyleConfig {
	fn new(selectable: bool) -> Self { Self { selectable, ..Self::default() } }
}
impl LineStyleConfig {
	fn egui_line_style(&self) -> egui_plot::LineStyle {
		match self.line_style {
			LineStyleType::Solid => egui_plot::LineStyle::Solid,
			LineStyleType::Dotted => egui_plot::LineStyle::Dotted { spacing: self.line_style_size },
			LineStyleType::Dashed => egui_plot::LineStyle::Dashed { length: self.line_style_size },
		}
	}
}

impl<T: EvalexprFloat> Entry<T> {
	pub fn symbol(&self) -> &'static str {
		match self.ty {
			EntryType::Function { .. } => "λ",
			EntryType::Constant { .. } => {
				if self.active {
					"⏸"
				} else {
					"⏵"
				}
			},
			EntryType::Points { .. } => "◊",
			EntryType::Folder { .. } => {
				if self.active {
					"📂"
				} else {
					"📁"
				}
			},
		}
	}
	pub fn symbol_with_name(&self) -> &'static str {
		match self.ty {
			EntryType::Function { .. } => "λ   Function",
			EntryType::Constant { .. } => "⏵ Constant",
			EntryType::Points { .. } => "◊ Points",
			EntryType::Folder { .. } => "📂 Folder",
		}
	}
	pub fn name(&self) -> &'static str {
		match self.ty {
			EntryType::Function { .. } => "Function",
			EntryType::Constant { .. } => "Constant",
			EntryType::Points { .. } => "Points",
			EntryType::Folder { .. } => "Folder",
		}
	}
	pub fn color(&self) -> Color32 { COLORS[self.color % NUM_COLORS] }
	pub fn new_function(id: u64, text: &str) -> Self {
		Self {
			id,
			color: id as usize % NUM_COLORS,
			active: true,
			name: String::new(),
			draw_buffer_scheduler: DrawBufferScheduler::new(),
			ty: EntryType::Function {
				identifier:   istr_empty(),
				can_be_drawn: true,

				func:                Expr::from_text(text),
				parametric:          false,
				parametric_fill:     false,
				range_start:         Expr::from_text("-2"),
				range_end:           Expr::from_text("2"),
				ty:                  FunctionType::Expression,
				fill_rule:           FillRule::default(),
				style:               LineStyleConfig::new(true),
				implicit_resolution: DEFAULT_IMPLICIT_RESOLUTION,
			},
		}
	}
	pub fn new_constant(id: u64) -> Self {
		Self {
			id,
			color: id as usize % NUM_COLORS,
			active: false,
			name: String::new(),
			draw_buffer_scheduler: DrawBufferScheduler::new(),
			ty: EntryType::Constant {
				istr_name:   istr_empty(),
				value:       T::ZERO,
				step:        0.01,
				ty:          ConstantType::LoopForwardAndBackward { forward: true },
				range_start: Expr::from_text("-10"),
				range_end:   Expr::from_text("10"),
			},
		}
	}
	pub fn new_points(id: u64) -> Self {
		Self {
			id,
			color: id as usize % NUM_COLORS,
			active: true,
			name: String::new(),
			draw_buffer_scheduler: DrawBufferScheduler::new(),
			ty: EntryType::Points {
				identifier: istr_empty(),
        points_ty: PointsType::Separate(vec![PointEntry::default()]),
				style:      PointStyle::default(),
			},
		}
	}
	pub fn new_folder(id: u64) -> Self {
		Self {
			id,
			color: id as usize % NUM_COLORS,
			active: true,
			name: String::new(),
			draw_buffer_scheduler: DrawBufferScheduler::new(),
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
pub struct PointDrag {
	pub drag_point:               Option<DragPoint>,
	pub drag_type:                PointDragType,
	pub both_drag_dirs_available: bool,
}
#[derive(Clone, Debug)]
pub struct PointEntry<T: EvalexprFloat> {
	pub x:    Expr<T>,
	pub y:    Expr<T>,
	pub drag: PointDrag,
	pub val:  Option<(T, T)>,
}

impl<T: EvalexprFloat> Default for PointEntry<T> {
	fn default() -> Self {
		Self {
			x:    Expr::default(),
			y:    Expr::default(),
			drag: PointDrag {
				drag_point:               None,
				drag_type:                PointDragType::default(),
				both_drag_dirs_available: true,
			},
			val:  None,
		}
	}
}
#[derive(Clone, Debug)]
pub enum PointsType<T: EvalexprFloat> {
	Separate(Vec<PointEntry<T>>),
	SingleExpr { expr: Expr<T>, val:Vec<(T,T)> }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConstantType {
	// LoopForwardAndBackward { start: f64, end: f64, forward: bool },
	// LoopForward { start: f64, end: f64 },
	// PlayOnce { start: f64, end: f64 },
	// PlayIndefinitely { start: f64 },
	LoopForwardAndBackward { forward: bool },
	LoopForward,
	PlayOnce,
	PlayIndefinitely,
}
impl ConstantType {
	pub fn range<T: EvalexprFloat>(
		&self, ctx: &HashMapContext<T>, start: &Expr<T>, end: &Expr<T>,
	) -> Result<RangeInclusive<f64>, String> {
		let mut stack = Stack::<T>::with_capacity(0);
		match self {
			ConstantType::LoopForwardAndBackward { .. }
			| ConstantType::LoopForward
			| ConstantType::PlayOnce => {
				let start = if let Some(start) = start.node.as_ref() {
					start.eval_float_with_context(&mut stack, ctx).map_err(|e| e.to_string())?.to_f64()
				} else {
					-f64::INFINITY
				};
				let end = if let Some(end) = end.node.as_ref() {
					end.eval_float_with_context(&mut stack, ctx).map_err(|e| e.to_string())?.to_f64()
				} else {
					f64::INFINITY
				};

				Ok(start.to_f64()..=end.to_f64())
			},
			ConstantType::PlayIndefinitely => {
				let start = if let Some(start) = start.node.as_ref() {
					start.eval_float_with_context(&mut stack, ctx).map_err(|e| e.to_string())?.to_f64()
				} else {
					-f64::NEG_INFINITY
				};
				Ok(start..=f64::INFINITY)
			},
		}
	}
	pub fn symbol(&self) -> &'static str {
		match self {
			ConstantType::LoopForwardAndBackward { .. } => "🔁",
			ConstantType::LoopForward => "🔂",
			ConstantType::PlayOnce => "⏯",
			ConstantType::PlayIndefinitely => "🔀",
		}
	}
	pub fn name(&self) -> &'static str {
		match self {
			ConstantType::LoopForwardAndBackward { .. } => "🔁 Loop forward and Backward",
			ConstantType::LoopForward => "🔂 Loop forward",
			ConstantType::PlayOnce => "⏭ Play once",
			ConstantType::PlayIndefinitely => "🔀 Play indefinitely",
		}
	}
}

pub static RESERVED_NAMES: [&str; 2] = ["x", "y"];

pub fn f64_to_value<T: EvalexprFloat>(x: f64) -> Value<T> { Value::<T>::Float(T::from_f64(x)) }
// pub fn f64_to_float<T: EvalexprFloat>(x: f64) -> T { T::f64_to_float(x) }

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
