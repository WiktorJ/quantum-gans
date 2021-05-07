(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("scrbook" "headsepline" "footsepline" "footinclude=false" "oneside" "fontsize=11pt" "paper=a4" "listof=totoc" "bibliography=totoc" "DIV=12")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("nag" "l2tabu" "orthodox")))
   (TeX-run-style-hooks
    "latex2e"
    "settings"
    "pages/cover"
    "pages/title"
    "pages/disclaimer"
    "pages/acknowledgements"
    "pages/abstract"
    "chapters/01_introduction"
    "chapters/02_quantum_computing_intro"
    "chapters/03_GANs"
    "chapters/05_quantum_gans"
    "chapters/06_my_contribution"
    "chapters/07_experiment_results"
    "chapters/08_conclusions"
    "pages/appendix"
    "nag"
    "scrbook"
    "scrbook10")
   (TeX-add-symbols
    "getUniversity"
    "getFaculty"
    "getTitle"
    "getTitleGer"
    "getAuthor"
    "getDoctype"
    "getSupervisor"
    "getSubmissionDate"
    "getSubmissionLocation"))
 :latex)

