use eframe::egui::{Area, DragValue, Frame, Id, Slider, SliderClamping, Ui, Widget};

pub fn duplicate_entry_btn(ui: &mut Ui, text: &str) -> bool {
	ui.button("🗐").on_hover_text(format!("Duplicate {text}")).clicked()
}
pub fn remove_entry_btn(ui: &mut Ui, text: &str) -> bool {
	ui.button("X").on_hover_text(format!("Remove {text}")).clicked()
}

pub fn popup_label(ui: &Ui, id: Id, label: String, pos: [f32; 2]) {
	Area::new(id).fixed_pos([pos[0] + 5.0, pos[1] + 5.0]).interactable(false).show(ui.ctx(), |ui| {
		Frame::popup(ui.style()).show(ui, |ui| {
			ui.horizontal(|ui| ui.label(label));
		});
	});
}

pub fn full_width_slider(
	ui: &mut Ui, value: &mut f64, range: core::ops::RangeInclusive<f64>, step: f64, eps: f64,
) -> bool {
	let mut changed = false;
	// ui.with_layout(egui::Layout::centered_and_justified(egui::Direction::RightToLeft), |ui| {
	ui.horizontal(|ui| {
		ui.set_min_width(80.0);
		let step = step.clamp(eps * 10.0, 1000.0) * 0.5;
		if DragValue::new(value).speed(step).ui(ui).changed() {
			changed = true;
		}
	});
	// if DragValue::new(value).speed(step * 0.5).ui(ui).changed() {
	// 	changed = true;
	// }
	let default_slider_width = ui.style().spacing.slider_width;
	ui.style_mut().spacing.slider_width = ui.available_width();
	if ui
		.add(Slider::new(value, range).step_by(step).clamping(SliderClamping::Never).show_value(false))
		.dragged()
	{
		changed = true;
	}
	ui.style_mut().spacing.slider_width = default_slider_width;
	// });
	changed
}
