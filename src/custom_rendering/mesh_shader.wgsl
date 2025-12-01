struct Locals {
    screen_size: vec2<f32>,
    rect_min: vec2<f32>,
    rect_size: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> r_locals: Locals;

fn position_from_screen(screen_pos: vec2<f32>) -> vec4<f32> {
    // Transform from screen space to rect-local space, then to clip space
    let local_pos = screen_pos - r_locals.rect_min;
    
    return vec4<f32>(
        2.0 * local_pos.x / r_locals.rect_size.x - 1.0,
        1.0 - 2.0 * local_pos.y / r_locals.rect_size.y,
        0.0,
        1.0,
    );
}

struct VertexInput {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}


fn unpack_color(color: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(color & 255u),
        f32((color >> 8u) & 255u),
        f32((color >> 16u) & 255u),
        f32((color >> 24u) & 255u),
    ) / 255.0;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = position_from_screen(in.pos);
    out.uv = in.uv;
    out.color = unpack_color(in.color);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
