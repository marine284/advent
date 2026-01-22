const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "_2024",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .omit_frame_pointer = true, // !!
        }),
    });

    const anycast =  b.dependency("anycast", .{ .target = target, .optimize = optimize }).module("anycast");
    exe.root_module.addImport("anycast", anycast);

    b.installArtifact(exe);

    // zig build asm  ->  zig-out/output.s  (feed to llvm-mca)
    const asm_step = b.step("asm", "Emit annotated asm for llvm-mca");
    const install_asm = b.addInstallFile(exe.getEmittedAsm(), "output.s");
    asm_step.dependOn(&install_asm.step);
    install_asm.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
}
