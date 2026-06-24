import os

with open('articel-v2/draft_iclr_2026.tex', 'r') as f:
    text = f.read()

start_idx = text.find('\\begin{abstract}')
end_idx = text.find('\\bibliography{')

body = text[start_idx:end_idx]

header = r"""\documentclass[10pt]{article} % For LaTeX2e
\usepackage[preprint]{tmlr}

\input{math_commands.tex}

\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{url}
\usepackage{tikz}
\usetikzlibrary{arrows}

\title{Is Spike-Driven Self-Attention Necessary? The Inefficiency\\ of Spike-Overlap Attention in Spiking\\ Sentence Embeddings}

\author{\name Muhammad Akhyar \email akhyarsafrudin@gmail.com}

\newcommand{\fix}{\marginpar{FIX}}
\newcommand{\new}{\marginpar{NEW}}

\def\month{06}  
\def\year{2026} 
\def\openreview{\url{}} 

\begin{document}

\maketitle

"""

footer = r"""
\bibliography{iclr2026_conference}
\bibliographystyle{tmlr}

\end{document}
"""

with open('tmlr-style-file-main/tmlr-style-file-main/main.tex', 'w') as f:
    f.write(header + body + footer)

