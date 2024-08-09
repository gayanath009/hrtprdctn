{pkgs}: {
  deps = [
    pkgs.emacsPackages.pickle
    pkgs.python312Packages.xgboost
    pkgs.python311Packages.scikit-learn
    pkgs.python312Packages.tensorflow-bin
    pkgs.python311Packages.pandas
    pkgs.python312Packages.scipy
    pkgs.python311Packages.numpy
    pkgs.python312Packages.gunicorn
    pkgs.python311Packages.flask
  ];
}
