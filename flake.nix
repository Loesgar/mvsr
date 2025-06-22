{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      perSystem =
        { pkgs, ... }:
        {
          packages = rec {
            default = mvsr;
            mvsr = pkgs.callPackage ./mvsr/package.nix { };
          };

          devShells = {
            default = pkgs.mkShell {
              packages = with pkgs; [ nixfmt-rfc-style ];
            };
            mvsr = import ./mvsr/shell.nix { inherit pkgs; };
            docs = import ./docs/shell.nix { inherit pkgs; };
            lang-python = import ./lang/python/shell.nix { inherit pkgs; };
          };
        };
    };
}
