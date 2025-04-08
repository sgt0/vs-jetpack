{
  description = "vs-jetpack devenv";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    vs-nix-overlay.url = "github:sgt0/vs-nix-overlay";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    vs-nix-overlay,
  }:
    flake-utils.lib.eachDefaultSystem
    (system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [
          vs-nix-overlay.overlays.default
        ];
      };
    in {
      devShells = {
        default =
          pkgs.mkShell
          {
            nativeBuildInputs = with pkgs; [
              python312
              uv

              (vapoursynth.withPlugins [
                vspkgs.vapoursynthPlugins.akarin_jet
                vspkgs.vapoursynthPlugins.resize2
                vspkgs.vapoursynthPlugins.vszip
              ])
            ];
          };
      };
    });
}
