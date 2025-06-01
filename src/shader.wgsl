var<push_constant> scale: vec2f;

struct VertexInput {
    @builtin(vertex_index) index: u32,
    @location(0) mat0: vec2f,
    @location(1) mat1: vec2f,
    @location(2) center: vec2f,
    @location(3) color: vec3f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
}

@vertex fn vertex(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    output.position = vec4f(scale * (mat2x2f(input.mat0, input.mat1) * array<vec2f, 4>(
        vec2f(1.0, 1.0),
        vec2f(1.0, -1.0),
        vec2f(-1.0, 1.0),
        vec2f(-1.0, -1.0),
    )[input.index] + input.center), 0.0, 1.0);

    output.color = input.color;
    return output;
}

struct FragmentInput {
    @location(0) color: vec3f,
}

struct FragmentOutput {
    @location(0) color: vec4f,
}

@fragment fn fragment(input: FragmentInput) -> FragmentOutput {
    var output: FragmentOutput;
    output.color = vec4f(input.color, 1.0);
    return output;
}
