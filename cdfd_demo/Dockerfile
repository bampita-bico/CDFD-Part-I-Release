FROM python:3.11-slim

WORKDIR /demo

RUN pip install --no-cache-dir numpy matplotlib

COPY . .

CMD ["sh", "-c", \
     "python paper_scripts/afl_principle.py && \
      python paper_scripts/tri_regime.py && \
      python paper_scripts/life_emergence.py && \
      echo 'Done. Figures: afl_principle.png  tri_regime.png  life_emergence.png'"]
