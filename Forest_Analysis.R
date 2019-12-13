setwd("Documents/UCSD/BMS/MED264/Group_Project/Formal_Analysis/")

inputs <- as.data.frame(as.matrix(read.csv("Final_Dataset/inputs_trunc_g3.csv", header=TRUE, sep=",",row.names=1, na.strings = "NA", stringsAsFactors = FALSE)))
outcomes <- read.csv("Final_Dataset/outcomes_trunc_g3.csv", header=TRUE, sep=",",row.names=1, na.strings = "NA", stringsAsFactors = FALSE)

inputs$year <- as.factor(inputs$year)

# Fit the full model 
full.model1 <- lm(outcomes$total_mortality ~ ., data=inputs)
summary(full.model1)
# Fit the full model 
full.model2 <- lm(outcomes$circulatory_mortality ~ ., data=inputs)
summary(full.model2)
# Fit the full model 
full.model3 <- lm(outcomes$respiratory_mortality ~ ., data=inputs)
summary(full.model3)
# Fit the full model 
full.model4 <- lm(outcomes$external_mortality ~ ., data=inputs)
summary(full.model4)
# Fit the full model 
full.model5 <- lm(outcomes$digestive_mortality ~ ., data=inputs)
summary(full.model5)
# Fit the full model 
full.model6 <- lm(outcomes$nervous_mortality ~ ., data=inputs)
summary(full.model6)
# Fit the full model 
full.model7 <- lm(outcomes$genitourinary_mortality ~ ., data=inputs)
summary(full.model7)
# Fit the full model 
full.model8 <- lm(outcomes$mental_mortality ~ ., data=inputs)
summary(full.model8)
# Fit the full model 
full.model9 <- lm(outcomes$cancer_mortality ~ ., data=inputs)
summary(full.model9)
# Fit the full model 
full.model10 <- lm(outcomes$infection_mortality ~ ., data=inputs)
summary(full.model10)

sink("lm.txt")
print(summary(full.model1))
print(summary(full.model2))
print(summary(full.model3))
print(summary(full.model4))
print(summary(full.model5))
print(summary(full.model6))
print(summary(full.model7))
print(summary(full.model8))
print(summary(full.model9))
print(summary(full.model10))
sink()  # returns output to the console


