


details = read.csv("./Ensimag/ProjetFilé/Kaggle/NCAA/dataTestPredictors.csv",header=T)

mylm <- lm(details$Wscore ~ .-Wfgm-Wfgm3-Wftm, data=details)
summary(mylm)