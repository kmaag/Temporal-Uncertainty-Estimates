args = commandArgs(trailingOnly=TRUE)

#install.packages("UBL")
library(MBA)
library(gstat)
library(sp)
library(automap)
library(randomForest) 
library(UBL)
library(RcppCNPy)
library(lattice)
library(grid)
library(DMwR)

path_Xa <- paste(args[2], "Xa_train_run",args[1],".npy",sep = "", collapse = NULL)
path_ya <- paste(args[2], "ya_train_run",args[1],".npy",sep = "", collapse = NULL)

Xa <- npyLoad(path_Xa)
ya <- npyLoad(path_ya)

Xa_ya <- cbind ( ya , Xa )
Xa_ya_frame <- as.data.frame(Xa_ya)

rel <- matrix(0, ncol = 2, nrow = 0)
rel <- rbind(rel, c(0.1, 0.5))
rel <- rbind(rel, c(0.6, 0.2))
rel <- rbind(rel, c(1, 0.2))

out_Xa_ya_frame <- SmoteRegress(ya~., Xa_ya_frame, dist="Euclidean",rel=rel,C.perc=list(0.5,2.5),k=5)

for(i in 1:8){
  if(i==1 || i==2){
    rel <- matrix(0, ncol = 2, nrow = 0)
    rel <- rbind(rel, c(0.1, 0.7))
    rel <- rbind(rel, c(0.6, 0.5))
    rel <- rbind(rel, c(1, 0.1))
  }
  else if(i==3 || i==4){
    rel <- matrix(0, ncol = 2, nrow = 0)
    rel <- rbind(rel, c(0.1, 0.9))
    rel <- rbind(rel, c(0.6, 0.5))
    rel <- rbind(rel, c(1, 0.3))
  }
  else if(i==5 || i==6){
    rel <- matrix(0, ncol = 2, nrow = 0)
    rel <- rbind(rel, c(0.1, 0.1))
    rel <- rbind(rel, c(0.6, 0.7))
    rel <- rbind(rel, c(1, 0.9))
  }
  else if(i==7 || i==8){
    rel <- matrix(0, ncol = 2, nrow = 0)
    rel <- rbind(rel, c(0.1, 0.2))
    rel <- rbind(rel, c(0.6, 0.5))
    rel <- rbind(rel, c(1, 0.8))
  }
  tmp_out_Xa_ya_frame <- SmoteRegress(ya~., Xa_ya_frame, dist="Euclidean",rel=rel,C.perc=list(0.5,2.5),k=5)
  out_Xa_ya_frame <- rbind ( out_Xa_ya_frame , tmp_out_Xa_ya_frame )
}

ya_new = out_Xa_ya_frame[,1]
num_metrics <- dim(out_Xa_ya_frame)[2]
Xa_new_frame = out_Xa_ya_frame[,c(2:num_metrics)]

save_path_Xa <- paste(args[2], "Xa_A_run",args[1],".npy",sep = "", collapse = NULL)
save_path_ya <- paste(args[2], "ya_A_run",args[1],".npy",sep = "", collapse = NULL)

Xa_new <- data.matrix(Xa_new_frame)
npySave(save_path_Xa, Xa_new)
npySave(save_path_ya, ya_new) 
