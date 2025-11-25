struct VertexInput {
    @location(0) position: vec2<f32>,
}
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}
struct Uniforms {
    color: vec4<f32>,
}
@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Vertices are already in -1..1 space
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    return out;
}
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return uniforms.color;
}

// fullscreen quad
@vertex
fn vs_quad(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0)
    );
    out.position = vec4<f32>(pos[vertex_index], 0.0, 1.0);
    return out;
}
@fragment
fn fs_quad() -> @location(0) vec4<f32> {
    return uniforms.color;
}
