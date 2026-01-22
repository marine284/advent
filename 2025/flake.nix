{
  inputs.nixpkgs.url = "nixpkgs";

  outputs = { nixpkgs, ... }:
    let
      forAllSystems = f: builtins.mapAttrs f nixpkgs.legacyPackages;
    in
    {
      devShells = forAllSystems (system: pkgs:
        let
          commonPkgs = [
            pkgs.cmake
            pkgs.ninja
            pkgs.clang-tools
            pkgs.editorconfig-checker
            pkgs.libllvm # llvm-mca
          ];
        in
        {
          # 1. CLANG (Default) - "nix develop"
          default = pkgs.mkShell.override { stdenv = pkgs.clangStdenv; } {
            packages = commonPkgs;
            shellHook = ''
              unset NIX_ENFORCE_NO_NATIVE
            '';
          };

          # 2. GNU (GCC) - "nix develop .#gnu"
          gnu = pkgs.mkShell.override { stdenv = pkgs.gccStdenv; } {
            # Standard mkShell uses GCC by default on Linux
            packages = commonPkgs;
            shellHook = ''
              unset NIX_ENFORCE_NO_NATIVE
            '';
          };
        });
    };
}
