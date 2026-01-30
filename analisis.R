# Librerías necesarias
install.packages(c("tidyverse", "knitr", "dplyr", "ggplot2", "MASS", "ggcorrplot", "fitdistrplus", "leaps", "glmnet", "caret"))
library(tidyverse)
library(knitr)
library(dplyr)
library(ggplot2)
library(MASS)
library(ggcorrplot)
library(fitdistrplus)
library(leaps)
library(glmnet)
library(caret)

# Cargar datos
df <- read_csv('df_precios_con_servicios (2).csv')

# Resumen y limpieza inicial
kable(summary(df))
nrow(df)
df <- df %>% dplyr::select(where(is.numeric))
head(df)

# NA por columna y total
colSums(is.na(df))
# sum(is.na(df))

summary(df)

# Histogramas de variables numéricas
df %>%
  pivot_longer(cols = where(is.numeric), names_to = "variable", values_to = "valor") %>%
  ggplot(aes(x = valor)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal()

# Boxplots de variables numéricas
df %>%
  pivot_longer(cols = where(is.numeric), names_to = "variable", values_to = "valor") %>%
  ggplot(aes(x = valor, y = variable)) +
  geom_boxplot() +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal()

# Matriz de dispersión
options(repr.plot.width=10, repr.plot.height=10)
pairs(df, main = "Scatterplot Matrix")

# Dispersión Precio_MXN vs otras variables
df %>%
  pivot_longer(cols = -Precio_MXN, names_to = "variable", values_to = "valor") %>%
  ggplot(aes(x = valor, y = Precio_MXN)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", col = "red", se = FALSE) +
  facet_wrap(~ variable, scales = "free_x", ncol = 3) +
  labs(title = "Dispersion Plots of Precio_MXN vs. Other Variables",
       x = "Variable Value",
       y = "Precio_MXN") +
  theme_minimal()

# Checkpoint
df_checkpoint1 <- df
df <- df_checkpoint1

# Filtrado de outliers
df <- df %>%
  filter(
    Precio_MXN >= quantile(Precio_MXN, 0.06, na.rm = TRUE),
    Precio_MXN <= quantile(Precio_MXN, 0.94, na.rm = TRUE),
    Superficie >= quantile(Superficie, 0.05, na.rm = TRUE),
    Superficie <= quantile(Superficie, 0.95, na.rm = TRUE),
    Recamaras <= quantile(Recamaras, 0.99, na.rm = TRUE),
    Baños <= quantile(Baños, 0.99, na.rm = TRUE),
    Estacionamientos <= quantile(Estacionamientos, 0.99, na.rm = TRUE),
    Esparcimiento <= quantile(Esparcimiento, 0.95, na.rm = TRUE),
    Hospitales <= quantile(Hospitales, 0.95, na.rm = TRUE),
    Escuelas <= quantile(Escuelas, 0.95, na.rm = TRUE),
    Restaurantes <= quantile(Restaurantes, 0.95, na.rm = TRUE),
    Carpetas <= quantile(Carpetas, 0.95, na.rm = TRUE),
    Transporte <= quantile(Transporte, 0.95, na.rm = TRUE),
    Parques <= quantile(Parques, 0.95, na.rm = TRUE)
  )

nrow(df)
summary(df)

# Histogramas y boxplots tras filtrado
df %>%
  pivot_longer(cols = where(is.numeric), names_to = "variable", values_to = "valor") %>%
  ggplot(aes(x = valor)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal()

df %>%
  pivot_longer(cols = where(is.numeric), names_to = "variable", values_to = "valor") %>%
  ggplot(aes(x = valor, y = variable)) +
  geom_boxplot() +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal()

df %>%
  pivot_longer(cols = -Precio_MXN, names_to = "variable", values_to = "valor") %>%
  ggplot(aes(x = valor, y = Precio_MXN)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", col = "red", se = FALSE) +
  facet_wrap(~ variable, scales = "free_x", ncol = 3) +
  labs(title = "Scatter Plot de Precio_MXN",
       x = "Variable Value",
       y = "Precio_MXN") +
  theme_minimal()

# Correlación
corr <- cor(df, use = "complete.obs", method = "pearson")
options(repr.plot.width = 10, repr.plot.height = 10)
ggcorrplot(corr,
           hc.order = TRUE,
           lab = TRUE,
           colors = c("blue", "white", "red"))

# Checkpoint
df_checkpoint2 <- df
df <- df_checkpoint2

# Histograma log(Precio_MXN) con normal
log_precio <- log(df$Precio_MXN)
mu_log  <- mean(log_precio)
sd_log  <- sd(log_precio)
ggplot(df, aes(x = log_precio)) +
  geom_histogram(
    aes(y = after_stat(density)),
    bins = 30,
    fill = "darkgreen",
    color = "white",
    alpha = 0.6
  ) +
  stat_function(
    fun = dnorm,
    args = list(mean = mu_log, sd = sd_log),
    color = "red",
    linewidth = 1
  ) +
  labs(
    title = "Histograma de log(Precio_MXN) con Normal",
    x = "log(Precio_MXN)",
    y = "Densidad"
  ) +
  annotate(
    "text",
    x = Inf, y = Inf,
    hjust = 1.1, vjust = 2,
    label = paste0(
      "μ = ", round(mu_log, 2),
      "\nσ = ", round(sd_log, 2)
    )
  ) +
  theme_minimal()

# Ajuste normal log(Precio_MXN)
x_log <- log(df$Precio_MXN)
fit_norm_log <- fitdist(x_log, "norm")
qqnorm(x_log)
qqline(x_log, col = "red")
fit_norm_log

# Regresión lineal múltiple
lm.log <- lm(
  log(Precio_MXN) ~ Estacionamientos + Superficie + Recamaras + Baños +
    Latitud + Longitud + Hospitales + Escuelas +
    Esparcimiento + Transporte + Parques,
  data = df
)
summary(lm.log)
par(mfrow=c(2,2)); plot(lm.log); par(mfrow=c(1,1))

# Mejor subconjunto
regfit.full <- regsubsets(Precio_MXN ~ ., data = df, nvmax = 14)
reg.summary <- summary(regfit.full)
names(reg.summary)
reg.summary$rsq

options(repr.plot.width=12, repr.plot.height=6)
par(mfrow = c(1, 2))
plot(reg.summary$rss, xlab = "N° de variables",
    ylab = "RSS", type = "l")
plot(reg.summary$adjr2, xlab = "N° de variables",
    ylab = "Adjusted RSq", type = "l")
which.max(reg.summary$adjr2)
points(11, reg.summary$adjr2[11], col = "red", cex = 2, pch = 20)
plot(reg.summary$cp, xlab = "N° de variables",
    ylab = "Cp", type = "l")
which.min(reg.summary$cp)
points(10, reg.summary$cp[10], col = "red", cex = 2, pch = 20)
which.min(reg.summary$bic)
plot(reg.summary$bic, xlab = "N° de variables",
    ylab = "BIC", type = "l")
points(6, reg.summary$bic[6], col = "red", cex = 2, pch = 20)
plot(regfit.full, scale = "r2")
plot(regfit.full, scale = "adjr2")
plot(regfit.full, scale = "Cp")
plot(regfit.full, scale = "bic")
coef(regfit.full, 6)

# Eliminación delantera y posterior
regfit.fwd <- regsubsets(Precio_MXN ~ ., data = df, nvmax = 14, method = "forward")
summary(regfit.fwd)
regfit.bwd <- regsubsets(Precio_MXN ~ ., data = df, nvmax = 14, method = "backward")
summary(regfit.bwd)
print(coef(regfit.full, 6))
print("\n")
print(coef(regfit.fwd, 6))
print("\n")
print(coef(regfit.bwd, 6))

# Variables óptimas
variables_optimas <- c("Precio_MXN", "Superficie", "Baños", "Hospitales", "Escuelas", "Esparcimiento", "Latitud")
df_optimo <- df[, variables_optimas]
head(df_optimo)
str(df_optimo)

# Checkpoint
df_checkpoint3 <- df
df <- df_checkpoint3
df <- df_optimo

# Ridge y Lasso
df <- df %>% drop_na()
x <- as.matrix(df %>% dplyr::select(-Precio_MXN))
y <- df$Precio_MXN

# Ridge
ridge <- glmnet(x, y, alpha = 0)
cv.ridge <- cv.glmnet(x, y, alpha = 0)
coef(cv.ridge, s = "lambda.min")

# Lasso
lasso <- glmnet(x, y, alpha = 1)
cv.lasso <- cv.glmnet(x, y, alpha = 1)
coef(cv.lasso, s = "lambda.min")

# Gráficas de coeficientes
plot(ridge, xvar = "lambda", label = TRUE, main = "Ridge: Coeficientes vs Lambda")
plot(lasso, xvar = "lambda", label = TRUE, main = "Lasso: Coeficientes vs Lambda")

# Validación cruzada
set.seed(0)
train_control <- trainControl(method = "cv", number = 10)
model_cv <- train(log(Precio_MXN) ~ ., data = df, method = "lm", trControl = train_control)
print(model_cv)
plot(cv.ridge, main = "Ridge: Validación cruzada")
plot(cv.lasso, main = "Lasso: Validación cruzada")

# Métricas Ridge y Lasso
df_clean <- df %>% drop_na()
x_cv <- as.matrix(df_clean %>% dplyr::select(-Precio_MXN))
y_cv <- df_clean$Precio_MXN
y_cv <- log(y_cv)

cat("--- Ridge (alpha = 0) con VC ---\n")
set.seed(123)
cv.ridge <- cv.glmnet(x_cv, y_cv, alpha = 0, family = "gaussian")
lambda_min_ridge <- cv.ridge$lambda.min
cat("lambda optimo para Ridge:", lambda_min_ridge, "\n")
rmse_ridge <- sqrt(cv.ridge$cvm[cv.ridge$lambda == lambda_min_ridge])
cat("RMSE Ridge:", rmse_ridge, "\n")
cat("Coeficientes de Ridge:\n")
print(coef(cv.ridge, s = "lambda.min"))

cat("\n--- Lasso (alpha = 1) con VC ---\n")
set.seed(123)
cv.lasso <- cv.glmnet(x_cv, y_cv, alpha = 1, family = "gaussian")
lambda_min_lasso <- cv.lasso$lambda.min
cat("lambda optimo para Lasso:", lambda_min_lasso, "\n")
rmse_lasso <- sqrt(cv.lasso$cvm[cv.lasso$lambda == lambda_min_lasso])
cat("RMSE Lasso:", rmse_lasso, "\n")
cat("Coeficientes de Lasso:\n")
print(coef(cv.lasso, s = "lambda.min"))

# R^2 Ridge y Lasso
predictions_ridge <- predict(cv.ridge, s = "lambda.min", newx = x_cv)
ss_total_ridge <- sum((y_cv - mean(y_cv))^2)
ss_residual_ridge <- sum((y_cv - predictions_ridge)^2)
r_squared_ridge <- 1 - (ss_residual_ridge / ss_total_ridge)
cat("R^2 para Ridge:", r_squared_ridge, "\n\n")

predictions_lasso <- predict(cv.lasso, s = "lambda.min", newx = x_cv)
ss_total_lasso <- sum((y_cv - mean(y_cv))^2)
ss_residual_lasso <- sum((y_cv - predictions_lasso)^2)
r_squared_lasso <- 1 - (ss_residual_lasso / ss_total_lasso)
cat("R^2 para Lasso:", r_squared_lasso, "\n")

# GLM Poisson
glm_model <- glm(Precio_MXN ~ ., data = df, family = poisson(link = "sqrt"))
summary(glm_model)
predicciones <- predict(glm_model, type = "response")
plot(df$Precio_MXN, predicciones,
     xlab = "Precio real",
     ylab = "Precio predicho",
     main = "GLM: Precio real vs predicho")
abline(0, 1, col = "red", lwd = 2)

mse <- sqrt(mean((df$Precio_MXN - predicciones)^2))
cat("RMSE:", mse, "\n")
mae <- mean(abs(df$Precio_MXN - predicciones))
cat("MAE:", mae, "\n")
residual_deviance <- glm_model$deviance
null_deviance <- glm_model$null.deviance
r2 <- (1 - (residual_deviance / null_deviance))
cat("Pseudo R^2:", r2, "\n")

# Residuos Poisson
residuos <- residuals(glm_model, type = "deviance")
hist(residuos, main = "Histograma de residuos", xlab = "Residuos", breaks = 30, col = "skyblue")
qqnorm(residuos, main = "QQ plot de residuos")
qqline(residuos, col = "red", lwd = 2)
plot(fitted(glm_model), residuos,
     xlab = "Valores ajustados",
     ylab = "Residuos",
     main = "Residuos vs Valores ajustados")
abline(h = 0, col = "red", lwd = 2)

# GLM Gamma
glm_model <- glm(Precio_MXN ~ ., data = df, family = Gamma(link = "sqrt"))
summary(glm_model)
predicciones <- predict(glm_model, type = "response")
plot(df$Precio_MXN, predicciones,
     xlab = "Precio real",
     ylab = "Precio predicho",
     main = "GLM: Precio real vs predicho")
abline(0, 1, col = "red", lwd = 2)

mse <- sqrt(mean((df$Precio_MXN - predicciones)^2))
cat("RMSE:", mse, "\n")
mae <- mean(abs(df$Precio_MXN - predicciones))
cat("MAE:", mae, "\n")
residual_deviance <- glm_model$deviance
null_deviance <- glm_model$null.deviance
r2 <- (1 - (residual_deviance / null_deviance))
cat("Pseudo R^2:", r2, "\n")

# Residuos Gamma
residuos <- residuals(glm_model, type = "deviance")
hist(residuos, main = "Histograma de residuos", xlab = "Residuos", breaks = 30, col = "skyblue")
qqnorm(residuos, main = "QQ plot de residuos")
qqline(residuos, col = "red", lwd = 2)
plot(fitted(glm_model), residuos,
     xlab = "Valores ajustados",
     ylab = "Residuos",
     main = "Residuos vs Valores ajustados")
abline(h = 0, col = "red", lwd = 2)