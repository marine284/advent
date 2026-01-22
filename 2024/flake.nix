{
  inputs = {
    nixpkgs.url = "nixpkgs";
    zig = {
      url = "github:silversquirl/zig-flake/compat";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    zls = {
      url = "github:zigtools/zls";
      #url = "github:zigtools/zls/0.15.0"; # pin
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.zig-overlay.follows = "zig";
    };
  };
  outputs =
    {
      nixpkgs,
      zig,
      zls,
      ...
    }:
    let
      forAllSystems = f: builtins.mapAttrs f nixpkgs.legacyPackages;
    in
    {
      devShells = forAllSystems (
        system: pkgs: {
          default = pkgs.mkShellNoCC {
            packages = [
              zig.packages.${system}."0.15.2" # pin
              zls.packages.${system}.zls
              pkgs.clang
              pkgs.libllvm # llvm-mca
            ];
          };
        }
      );
    };
}
