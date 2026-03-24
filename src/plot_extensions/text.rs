use core::ops::RangeInclusive;

use eframe::{egui::{Align2, Color32, Id, Shape, Stroke, StrokeKind, TextStyle, TextWrapMode, Ui, WidgetText}, epaint::TextShape};
use egui_plot::{PlotBounds, PlotGeometry, PlotItem, PlotItemBase, PlotPoint, PlotTransform};

/// Text inside the plot.
#[derive(Clone)]
pub struct TextPlotItem {
	name:         String,
	id:           Id,
	pub text:     WidgetText,
	pub position: PlotPoint,
	pub color:    Color32,
	pub anchor:   Align2,
	highlight:    bool,
	allow_hover:  bool,
	angle:        (f32, Align2),
}

impl TextPlotItem {
	pub fn new(name: impl Into<String>, position: PlotPoint, text: impl Into<WidgetText>) -> Self {
		let name = name.into();
		Self {
			text: text.into(),
			id: Id::new(&name),
			name,
			position,
			color: Color32::TRANSPARENT,
			anchor: Align2::CENTER_CENTER,
			allow_hover: true,
			highlight: false,
			angle: (0.0, Align2::CENTER_CENTER),
		}
	}

	/// Text color.
	#[inline]
	pub fn with_color(mut self, color: impl Into<Color32>) -> Self {
		self.color = color.into();
		self
	}

	pub fn with_angle(mut self, angle: f32, offset: Align2) -> Self {
		self.angle = (angle, offset);
		self
	}

	/// Anchor position of the text. Default is `Align2::CENTER_CENTER`.
	#[inline]
	pub fn _anchor(mut self, anchor: Align2) -> Self {
		self.anchor = anchor;
		self
	}

	/// Name of this plot item.
	///
	/// This name will show up in the plot legend, if legends are turned on.
	///
	/// Setting the name via this method does not change the item's id, so you can use it to
	/// change the name dynamically between frames without losing the item's state. You should
	/// make sure the name passed to [`Self::new`] is unique and stable for each item, or
	/// set unique and stable ids explicitly via [`Self::id`].
	#[inline]
	pub fn _with_name(mut self, name: impl ToString) -> Self {
		self.name = name.to_string();
		self.id = Id::new(&self.name);
		self
	}

	/// Highlight this plot item, typically by scaling it up.
	///
	/// If false, the item may still be highlighted via user interaction.
	#[inline]
	pub fn _with_highlight(mut self, highlight: bool) -> Self {
		self.highlight = highlight;
		self
	}

	/// Allowed hovering this item in the plot. Default: `true`.
	#[inline]
	pub fn _with_allow_hover(mut self, hovering: bool) -> Self {
		self.allow_hover = hovering;
		self
	}

	/// Sets the id of this plot item.
	///
	/// By default the id is determined from the name passed to [`Self::new`], but it can be
	/// explicitly set to a different value.
	#[inline]
	pub fn _with_id(mut self, id: impl Into<Id>) -> Self {
		self.id = id.into();
		self
	}
}

impl PlotItem for TextPlotItem {
	fn shapes(&self, ui: &Ui, transform: &PlotTransform, shapes: &mut Vec<Shape>) {
		let color =
			if self.color == Color32::TRANSPARENT { ui.style().visuals.text_color() } else { self.color };

		let galley = self.text.clone().into_galley(
			ui,
			Some(TextWrapMode::Extend),
			f32::INFINITY,
			TextStyle::Small,
		);

		let pos = transform.position_from_point(&self.position);
		let rect = self.anchor.anchor_size(pos, galley.size());

		shapes.push(
			TextShape::new(rect.min, galley, color).with_angle_and_anchor(self.angle.0, self.angle.1).into(),
		);

		if self.highlight {
			shapes.push(Shape::rect_stroke(
				rect.expand(1.0),
				1.0,
				Stroke::new(0.5, color),
				StrokeKind::Outside,
			));
		}
	}

	fn initialize(&mut self, _x_range: RangeInclusive<f64>) {}

	fn color(&self) -> Color32 { self.color }

	fn geometry(&self) -> PlotGeometry<'_> { PlotGeometry::None }

	fn bounds(&self) -> PlotBounds {
		let mut bounds = PlotBounds::NOTHING;
		bounds.extend_with(&self.position);
		bounds
	}

	fn base(&self) -> &PlotItemBase { unreachable!() }

	fn base_mut(&mut self) -> &mut PlotItemBase { unreachable!() }

	fn name(&self) -> &str { &self.name }

	fn highlight(&mut self) { self.highlight = true; }

	fn highlighted(&self) -> bool { self.highlight }

	fn allow_hover(&self) -> bool { self.allow_hover }

	fn id(&self) -> Id { self.id }
}
