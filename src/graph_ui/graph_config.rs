use eframe::egui::{self, DragValue, Grid};
use egui_plot::{PlotBounds, PlotTransform};
use serde::{Deserialize, Serialize};

use crate::AppConfig;

fn default_true() -> bool { true }
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GraphConfig {
	pub graph_plot_bounds: GraphPlotBounds,
	#[serde(default = "default_true")]
	pub show_legend:       bool,
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
			show_legend:       true,
			allow_zoom:        [true, true],
			allow_scroll:      [true, true],
			show_axes:         [true, true],
			show_grid:         [true, true],
			x_axis_label:      String::new(),
			y_axis_label:      String::new(),
			clamp_grid:        false,
			grid_spacing:      (8.0, 300.0),
			invert_axes:       [false, false],
		}
	}
}
impl GraphConfig {
	pub fn ui(&mut self, ui: &mut egui::Ui, config: &mut AppConfig) {
		Grid::new("graph_config").num_columns(2).striped(true).show(ui, |ui| {
			ui.label("Show Legend");
			ui.checkbox(&mut self.show_legend, "");
			ui.end_row();

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
	pub fn calc_plot_bounds(&self, view_aspect: f64) -> PlotBounds {
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
		self.data_aspect = (graph_bounds.width() / graph_bounds.height()) / view_aspect;
		self != &prev
	}
}
