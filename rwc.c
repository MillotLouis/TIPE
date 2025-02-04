#include <stdio.h>
#include <stdlib.h>
//affiche le contenu d'un fichier
void cat(char * filename) {
    char c;
    FILE * file = fopen(filename, "r");
    while (fscanf(file, "%c", &c) != EOF)
    {
        printf("%c", c);
    }
        printf("\n");
    fclose(file);
}

int ** create_matrix(char* nom, int * nbl, int * nbc) {
    // char c;
    FILE *file = fopen(nom, "r");
    fscanf(file,"%d", nbl);
    fscanf(file,"%d", nbc);
    
    int **mat = malloc(*nbl * sizeof(int*));
    for (int i = 0; i < *nbl; i++)
    {
        mat[i] = malloc(*nbc*sizeof(int));
        for(int j=0; j< *nbc;j++) 
        {
            fscanf(file,"%d",&(mat[i][j]));
        }
    }

    fclose(file);
        return mat;
    }

    void affichemat(int **mat, int nbl, int nbc) {
        for (int i = 0; i < nbl; i++) {
            for (int j = 0; j < nbc; j++) {
                printf("%d ", mat[i][j]);
            }
            printf("\n");
        }
    }
    

void transpose(char * nom, int ** tab, int nbl, int nbc) {
    FILE *file = fopen(nom,"w");
    for (int i = 0; i < nbc; i++)
    {
        for (int j = 0; j < nbl; j++)
        {
            fprintf(file,"%d ",tab[j][i]);
        }
        fprintf(file,"\n");
    }
}

/* void traiter (int argc, char** argv) {
    assert(argc <= 2);
    
} */


int main(int argc, char * argv[]) {
    // cat(argv[1]);

    int nbl,nbc;
    int **e = create_matrix(argv[1],&nbl,&nbc);
    printf("%d,\t%d\n",nbl,nbc); 
    affichemat(e,nbl,nbc);
    
    transpose(argv[2],e,nbl,nbc);

    for (int i = 0; i < nbl; i++) {
        free(e[i]);
    }
    free(e);
    return 0;
}
