{
  inputs = {
    nixpkgs.url = "github:cachix/devenv-nixpkgs/rolling";
    flake-utils.url = "github:numtide/flake-utils";
    vs-nix-overlay.url = "github:sgt0/vs-nix-overlay";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    vs-nix-overlay,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [vs-nix-overlay.overlays.default];
        };

        mkShellForPython = python:
          pkgs.mkShell {
            packages = with pkgs; [
              python
              uv

              (vspkgs.vapoursynth_73.withPlugins [
                vspkgs.vapoursynthPlugins.akarin_jet
                vspkgs.vapoursynthPlugins.resize2
                vspkgs.vapoursynthPlugins.vszip
              ])
            ];

            env =
              {
                UV_PYTHON_DOWNLOADS = "never";
                UV_PYTHON_PREFERENCE = "only-system";
              }
              // (pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
                LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
              });

            shellHook = ''
              uv sync --all-extras --locked
              . .venv/bin/activate
            '';
          };
      in {
        devShells.default = mkShellForPython pkgs.python312;
        devShells.python312 = mkShellForPython pkgs.python312;
        devShells.python314 = mkShellForPython pkgs.python314;
      }
    );
}
