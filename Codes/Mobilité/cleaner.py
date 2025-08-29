import os
import glob
import gzip
import shutil

def main():
    cwd = os.getcwd()

    # 1) Supprimer tous les fichiers rw*.params
    for f in glob.glob("rw*.params"):
        print(f"Suppression {f}")
        os.remove(f)

    # 2) Décompresser tous les rw*.movements.gz
    for gzfile in glob.glob("rw*.movements.gz"):
        base = gzfile[:-3]  # retire .gz → ex: rw1.movements
        out = os.path.join(cwd, os.path.basename(base))
        print(f"Extraction {gzfile} → {out}")

        with gzip.open(gzfile, "rb") as f_in, open(out, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        # 3) Supprimer le fichier .gz après extraction
        os.remove(gzfile)
        print(f"Supprimé {gzfile}")

    print("Terminé.")

if __name__ == "__main__":
    main()
