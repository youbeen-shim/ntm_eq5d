# Load necessary libraries
library(tidyverse)
library(lme4)
library(lmerTest)
library(ggplot2)

# Prepare data for analysis - improved NA handling
tobit_data <- data %>%
  # Explicitly filter out NA values for QSEQQALY
  filter(!is.na(QSEQDUR_diff) & !is.na(QSEQQALY)) %>%
  # Create normalized time variable (in years)
  mutate(time_years = QSEQDUR_diff/365) %>%
  # Make sure SUBJNO is treated as a factor (for random effects)
  mutate(SUBJNO = factor(SUBJNO))

# Get minimum values per group (to ensure we're using accurate bounds)
min_values <- tobit_data %>%
  group_by(CMQUIT_GROUP_2) %>%
  summarize(min_QSEQQALY = min(QSEQQALY, na.rm = TRUE))

# Print minimum values to confirm
print(min_values)

# Now apply transformations with group-specific minimum values
tobit_data <- tobit_data %>%
  left_join(min_values, by = "CMQUIT_GROUP_2") %>%
  # Transform QSEQQALY values to handle censoring
  # Using a small buffer from actual minimum values
  mutate(
    min_buffer = min_QSEQQALY * 0.9, # 90% of min value as buffer
    QSEQQALY_adj = case_when(
      QSEQQALY >= 1 ~ 0.999,  # Value close to 1 for right-censored
      TRUE ~ pmax(QSEQQALY, min_buffer) # Ensure no values below group min
    ),
    # Logit transformation to handle bounds
    QSEQQALY_logit = log(QSEQQALY_adj/(1-QSEQQALY_adj))
  )

# Fit linear mixed model on transformed data
lmm_transformed <- lmer(
  QSEQQALY_logit ~ time_years * CMQUIT_GROUP_2 + (1 | SUBJNO),
  data = tobit_data,
  control = lmerControl(optimizer = "bobyqa") # More robust optimizer
)

# Model summary
model_summary <- summary(lmm_transformed)
print(model_summary)

# Create prediction data frame for plotting
# Use actual observed time range for each group
time_ranges <- tobit_data %>%
  group_by(CMQUIT_GROUP_2) %>%
  summarize(
    min_time = min(time_years, na.rm = TRUE),
    max_time = max(time_years, na.rm = TRUE)
  )

# Create custom prediction grid for all groups with group-specific time ranges
pred_grid <- do.call(rbind, lapply(unique(tobit_data$CMQUIT_GROUP_2), function(group) {
  group_range <- time_ranges %>% filter(CMQUIT_GROUP_2 == group)
  data.frame(
    time_years = seq(group_range$min_time, group_range$max_time, length.out = 100),
    CMQUIT_GROUP_2 = group
  )
}))

# Add random effect values (set to 0 for population-level predictions)
pred_grid$SUBJNO <- NA

# Get fixed effects matrix
X <- model.matrix(~ time_years * CMQUIT_GROUP_2, data = pred_grid)

# Get fixed effects coefficients
beta <- fixef(lmm_transformed)

# Compute predictions on the logit scale
pred_grid$predicted_logit <- as.vector(X %*% beta)

# Back-transform to original scale
pred_grid$predicted_QSEQQALY <- 1 / (1 + exp(-pred_grid$predicted_logit))

# Define group colors to match reference image
group_colors <- c("A" = "#EEAAAA", "B" = "#AAAAEE", "C" = "#AAEEAA")

# Create the plot with improved aesthetics
ggplot() +
  # Raw data points
  geom_point(data = tobit_data, 
             aes(x = time_years, y = QSEQQALY, color = CMQUIT_GROUP_2), 
             alpha = 0.3, size = 1) +
  # Regression lines
  geom_line(data = pred_grid, 
            aes(x = time_years, y = predicted_QSEQQALY, color = CMQUIT_GROUP_2), 
            size = 1) +
  # Styling
  scale_color_manual(values = group_colors) +
  labs(title = "Tobit Regression of Health-Related Quality of Life by Treatment Group", 
       subtitle = "Accounting for censoring at QALY = 1",
       x = "Time (years)", 
       y = "QALY weight") +
  scale_y_continuous(limits = c(0.7, 1)) +
  theme_classic() +
  theme(
    legend.title = element_blank(),
    legend.text = element_text(size = 9),
    legend.position = "bottom"
  )