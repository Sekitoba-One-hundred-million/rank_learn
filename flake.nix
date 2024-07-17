{
  description = "Description for the project";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-python.url = "github:cachix/nixpkgs-python";

    devenv = {
      url = "github:cachix/devenv/python-rewrite";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.poetry2nix.follows = "poetry2nix";
    };

    devenv-root = {
      url =  "file+file:///dev/null";
      flake = false;
    };
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.devenv.flakeModule
        inputs.openmpi.flakeModule
      ];
      
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      
      perSystem = { config, self', inputs', pkgs, system, ... }: {
        
      };
      flake = {
      };
    };
}
