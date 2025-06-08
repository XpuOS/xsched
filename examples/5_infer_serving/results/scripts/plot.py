import os
import sys
import subprocess
from pathlib import Path


if __name__ == "__main__":
    raw_dir = Path(__file__).parent / "../raw"
    plot_dir = Path(__file__).parent / "../plot"

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    gnuplot_script = Path(__file__).parent / "triton.plt"

    output_eps = plot_dir / "fig13a.eps"
    output_pdf = plot_dir / "fig13a.pdf"

    standalone_data = raw_dir / "bert-high.standalone.txt"
    triton_data = raw_dir / "bert-high.triton.txt"
    triton_priority_data = raw_dir / "bert-high.triton-p.txt"
    xsched_data = raw_dir / "bert-high.xsched.txt"

    subprocess.run(["gnuplot",
                    "-e", f"eps_file='{output_eps}'",
                    "-e", f"standalone_data='{standalone_data}'",
                    "-e", f"triton_data='{triton_data}'",
                    "-e", f"triton_priority_data='{triton_priority_data}'",
                    "-e", f"xsched_data='{xsched_data}'",
                    gnuplot_script])
    
    subprocess.run(["epstopdf", output_eps, "--outfile", output_pdf])

    print(f"Plot saved to {output_eps} and {output_pdf}")
            
