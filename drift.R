# CNN-CAD Bayesian Performance Monitoring Framework
# 7-Year Simulation with Monthly Updates

library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(scales)

set.seed(42)

# ============================================================================
# CORE PARAMETERS AND FUNCTIONS
# ============================================================================

# Baseline assumptions
CANCER_PREVALENCE <- 0.005  # 0.5% cancer rate
N_SCANS_MONTHLY <- 1000
N_MONTHS <- 84  # 7 years

# Performance baselines
DUAL_SENSITIVITY <- 0.90
DUAL_SPECIFICITY <- 0.966  # 3.4% recall rate

# Cost structure (monthly, per 1000 scans)
COST_FIXED_AI <- 15000
COST_SINGLE_RADIOLOGIST <- 12500
COST_DUAL_RADIOLOGIST <- 25000
COST_MISSED_CANCER <- 200000
COST_FALSE_POSITIVE <- 153

# Performance thresholds
SENSITIVITY_THRESHOLD <- 0.85
SPECIFICITY_THRESHOLD <- 0.95

# Bayesian priors (conservative)
SENS_PRIOR_ALPHA <- 85
SENS_PRIOR_BETA <- 15
SPEC_PRIOR_ALPHA <- 966
SPEC_PRIOR_BETA <- 34

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

# Simulate CNN-CAD performance drift over time
simulate_cnn_performance <- function(months) {
  # Gradual drift model with some noise
  time_factor <- seq(0, 1, length.out = months)
  
  # Sensitivity: starts at 92%, gradually drifts to 86%
  base_sensitivity <- 0.92 - 0.06 * time_factor
  sensitivity <- pmax(0.80, pmin(0.95, base_sensitivity + rnorm(months, 0, 0.02)))
  
  # Specificity: starts at 97%, gradually drifts to 94%
  base_specificity <- 0.97 - 0.03 * time_factor  
  specificity <- pmax(0.90, pmin(0.99, base_specificity + rnorm(months, 0, 0.01)))
  
  return(data.frame(
    month = 1:months,
    true_sensitivity = sensitivity,
    true_specificity = specificity
  ))
}

# Generate monthly observations based on true performance
generate_monthly_data <- function(true_sens, true_spec, n_scans = N_SCANS_MONTHLY) {
  n_cancers <- rbinom(1, n_scans, CANCER_PREVALENCE)
  n_normals <- n_scans - n_cancers
  
  # Generate true outcomes
  detected_cancers <- rbinom(1, n_cancers, true_sens)
  correct_normals <- rbinom(1, n_normals, true_spec)
  
  return(list(
    n_cancers = n_cancers,
    n_normals = n_normals,
    detected_cancers = detected_cancers,
    correct_normals = correct_normals
  ))
}

# Update Bayesian posteriors
update_bayesian_posteriors <- function(sens_alpha, sens_beta, spec_alpha, spec_beta,
                                       n_cancers, detected_cancers, n_normals, correct_normals) {
  # Update sensitivity posterior
  new_sens_alpha <- sens_alpha + detected_cancers
  new_sens_beta <- sens_beta + n_cancers - detected_cancers
  
  # Update specificity posterior
  new_spec_alpha <- spec_alpha + correct_normals
  new_spec_beta <- spec_beta + n_normals - correct_normals
  
  return(list(
    sens_alpha = new_sens_alpha,
    sens_beta = new_sens_beta,
    spec_alpha = new_spec_alpha,
    spec_beta = new_spec_beta
  ))
}

# Calculate monthly costs
calculate_monthly_costs <- function(expected_sens, expected_spec) {
  expected_cancers <- CANCER_PREVALENCE * N_SCANS_MONTHLY
  expected_normals <- (1 - CANCER_PREVALENCE) * N_SCANS_MONTHLY
  
  # AI-CAD costs
  ai_missed_cancers <- expected_cancers * (1 - expected_sens)
  ai_false_positives <- expected_normals * (1 - expected_spec)
  ai_total_cost <- COST_FIXED_AI + COST_SINGLE_RADIOLOGIST + 
    ai_missed_cancers * COST_MISSED_CANCER + 
    ai_false_positives * COST_FALSE_POSITIVE
  
  # Dual radiologist costs
  dual_missed_cancers <- expected_cancers * (1 - DUAL_SENSITIVITY)
  dual_false_positives <- expected_normals * (1 - DUAL_SPECIFICITY)
  dual_total_cost <- COST_DUAL_RADIOLOGIST + 
    dual_missed_cancers * COST_MISSED_CANCER + 
    dual_false_positives * COST_FALSE_POSITIVE
  
  return(list(
    ai_cost = ai_total_cost,
    dual_cost = dual_total_cost,
    cost_difference = ai_total_cost - dual_total_cost
  ))
}

# Make decision based on multiple criteria
make_decision <- function(sens_alpha, sens_beta, spec_alpha, spec_beta) {
  # Expected performance
  expected_sens <- sens_alpha / (sens_alpha + sens_beta)
  expected_spec <- spec_alpha / (spec_alpha + spec_beta)
  
  # 80% Confidence intervals (10th and 90th percentiles)
  sens_ci_lower <- qbeta(0.1, sens_alpha, sens_beta)
  spec_ci_lower <- qbeta(0.1, spec_alpha, spec_beta)
  
  # Calculate costs
  costs <- calculate_monthly_costs(expected_sens, expected_spec)
  
  # Decision criteria
  cost_favorable <- costs$cost_difference < 0
  sensitivity_adequate <- sens_ci_lower > SENSITIVITY_THRESHOLD
  specificity_adequate <- spec_ci_lower > SPECIFICITY_THRESHOLD
  
  continue_ai <- cost_favorable && sensitivity_adequate && specificity_adequate
  
  return(list(
    decision = ifelse(continue_ai, "Continue AI-CAD", "Switch to Dual"),
    expected_sens = expected_sens,
    expected_spec = expected_spec,
    sens_ci_lower = sens_ci_lower,
    spec_ci_lower = spec_ci_lower,
    cost_difference = costs$cost_difference,
    ai_cost = costs$ai_cost,
    dual_cost = costs$dual_cost,
    sensitivity_risk = sens_ci_lower < 0.87,
    specificity_risk = spec_ci_lower < 0.96
  ))
}

# ============================================================================
# MAIN SIMULATION
# ============================================================================

# Run 7-year simulation
run_simulation <- function() {
  # Generate true performance trajectory
  true_performance <- simulate_cnn_performance(N_MONTHS)
  
  # Initialize results storage
  results <- data.frame(
    month = integer(N_MONTHS),
    true_sensitivity = numeric(N_MONTHS),
    true_specificity = numeric(N_MONTHS),
    estimated_sensitivity = numeric(N_MONTHS),
    estimated_specificity = numeric(N_MONTHS),
    sens_ci_lower = numeric(N_MONTHS),
    sens_ci_upper = numeric(N_MONTHS),
    spec_ci_lower = numeric(N_MONTHS),
    spec_ci_upper = numeric(N_MONTHS),
    cost_difference = numeric(N_MONTHS),
    decision = character(N_MONTHS),
    sensitivity_risk = logical(N_MONTHS),
    specificity_risk = logical(N_MONTHS),
    stringsAsFactors = FALSE
  )
  
  # Initialize Bayesian priors
  sens_alpha <- SENS_PRIOR_ALPHA
  sens_beta <- SENS_PRIOR_BETA
  spec_alpha <- SPEC_PRIOR_ALPHA
  spec_beta <- SPEC_PRIOR_BETA
  
  # Monthly simulation loop
  for (month in 1:N_MONTHS) {
    # Get true performance for this month
    true_sens <- true_performance$true_sensitivity[month]
    true_spec <- true_performance$true_specificity[month]
    
    # Generate monthly observations
    monthly_data <- generate_monthly_data(true_sens, true_spec)
    
    # Update Bayesian posteriors
    posteriors <- update_bayesian_posteriors(
      sens_alpha, sens_beta, spec_alpha, spec_beta,
      monthly_data$n_cancers, monthly_data$detected_cancers,
      monthly_data$n_normals, monthly_data$correct_normals
    )
    
    sens_alpha <- posteriors$sens_alpha
    sens_beta <- posteriors$sens_beta
    spec_alpha <- posteriors$spec_alpha
    spec_beta <- posteriors$spec_beta
    
    # Make decision based on current posteriors
    decision_result <- make_decision(sens_alpha, sens_beta, spec_alpha, spec_beta)
    
    # Store results
    results[month, ] <- list(
      month = month,
      true_sensitivity = true_sens,
      true_specificity = true_spec,
      estimated_sensitivity = decision_result$expected_sens,
      estimated_specificity = decision_result$expected_spec,
      sens_ci_lower = decision_result$sens_ci_lower,
      sens_ci_upper = qbeta(0.9, sens_alpha, sens_beta),
      spec_ci_lower = decision_result$spec_ci_lower,
      spec_ci_upper = qbeta(0.9, spec_alpha, spec_beta),
      cost_difference = decision_result$cost_difference,
      decision = decision_result$decision,
      sensitivity_risk = decision_result$sensitivity_risk,
      specificity_risk = decision_result$specificity_risk
    )
  }
  
  return(results)
}

# Run the simulation
simulation_results <- run_simulation()

# Add helper columns for visualization
simulation_results <- simulation_results %>%
  mutate(
    year = ceiling(month / 12),
    quarter = ceiling(month / 3),
    decision_numeric = ifelse(decision == "Continue AI-CAD", 1, 0),
    any_risk = sensitivity_risk | specificity_risk
  )

# Print summary statistics
cat("=== CNN-CAD 7-Year Simulation Summary ===\n")
cat("Total months simulated:", N_MONTHS, "\n")
cat("Months with AI-CAD continued:", sum(simulation_results$decision_numeric), "\n")
cat("Months switched to dual radiologists:", sum(1 - simulation_results$decision_numeric), "\n")
cat("First switch month:", ifelse(any(!simulation_results$decision_numeric), 
                                  min(which(!simulation_results$decision_numeric)), "Never"), "\n")
cat("Average monthly cost difference (AI - Dual): $", 
    round(mean(simulation_results$cost_difference), 0), "\n")

# Preview first 12 months
cat("\n=== First 12 Months Preview ===\n")
print(simulation_results[1:12, c("month", "true_sensitivity", "estimated_sensitivity", 
                                 "cost_difference", "decision")])


# ============================================================================
# VISUALIZATION 1: PERFORMANCE TRACKING WITH DECISION OVERLAY
# ============================================================================

create_performance_tracking_plot <- function(results) {
  # Prepare data for sensitivity plot
  sens_data <- results %>%
    select(month, true_sensitivity, estimated_sensitivity, 
           sens_ci_lower, sens_ci_upper, decision, sensitivity_risk) %>%
    mutate(year = month / 12)
  
  # Prepare data for specificity plot
  spec_data <- results %>%
    select(month, true_specificity, estimated_specificity, 
           spec_ci_lower, spec_ci_upper, decision, specificity_risk) %>%
    mutate(year = month / 12)
  
  # Create sensitivity plot
  p1 <- ggplot(sens_data, aes(x = year)) +
    # Confidence interval ribbon
    geom_ribbon(aes(ymin = sens_ci_lower, ymax = sens_ci_upper), 
                alpha = 0.3, fill = "steelblue") +
    # True sensitivity line
    geom_line(aes(y = true_sensitivity, color = "True Sensitivity"), 
              size = 1.2, alpha = 0.7) +
    # Estimated sensitivity line
    geom_line(aes(y = estimated_sensitivity, color = "Bayesian Estimate"), 
              size = 1) +
    # Threshold line
    geom_hline(yintercept = SENSITIVITY_THRESHOLD, 
               linetype = "dashed", color = "red", alpha = 0.7) +
    # Decision points
    geom_point(aes(y = estimated_sensitivity, 
                   shape = decision, 
                   fill = ifelse(sensitivity_risk, "Risk", "Safe")),
               size = 2, alpha = 0.8) +
    scale_color_manual(values = c("True Sensitivity" = "darkred", 
                                  "Bayesian Estimate" = "steelblue")) +
    scale_shape_manual(values = c("Continue AI-CAD" = 21, "Switch to Dual" = 24)) +
    scale_fill_manual(values = c("Risk" = "orange", "Safe" = "lightgreen")) +
    scale_y_continuous(labels = percent_format(), limits = c(0.8, 1.0)) +
    labs(
      title = "Sensitivity Tracking: CNN-CAD vs Dual Radiologist Threshold",
      subtitle = "Bayesian estimates with 80% confidence intervals",
      x = "Years",
      y = "Sensitivity (Cancer Detection Rate)",
      color = "Performance",
      shape = "Monthly Decision",
      fill = "Risk Status"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 11, color = "gray60"),
      legend.position = "right",
      panel.grid.minor = element_blank()
    ) +
    annotate("text", x = 6, y = SENSITIVITY_THRESHOLD + 0.01, 
             label = "85% Threshold", color = "red", size = 3)
  
  # Create specificity plot  
  p2 <- ggplot(spec_data, aes(x = year)) +
    # Confidence interval ribbon
    geom_ribbon(aes(ymin = spec_ci_lower, ymax = spec_ci_upper), 
                alpha = 0.3, fill = "darkgreen") +
    # True specificity line
    geom_line(aes(y = true_specificity, color = "True Specificity"), 
              size = 1.2, alpha = 0.7) +
    # Estimated specificity line
    geom_line(aes(y = estimated_specificity, color = "Bayesian Estimate"), 
              size = 1) +
    # Threshold line
    geom_hline(yintercept = SPECIFICITY_THRESHOLD, 
               linetype = "dashed", color = "red", alpha = 0.7) +
    # Dual radiologist baseline
    geom_hline(yintercept = DUAL_SPECIFICITY, 
               linetype = "dotted", color = "purple", alpha = 0.7) +
    # Decision points
    geom_point(aes(y = estimated_specificity, 
                   shape = decision, 
                   fill = ifelse(specificity_risk, "Risk", "Safe")),
               size = 2, alpha = 0.8) +
    scale_color_manual(values = c("True Specificity" = "darkgreen", 
                                  "Bayesian Estimate" = "darkgreen")) +
    scale_shape_manual(values = c("Continue AI-CAD" = 21, "Switch to Dual" = 24)) +
    scale_fill_manual(values = c("Risk" = "orange", "Safe" = "lightgreen")) +
    scale_y_continuous(labels = percent_format(), limits = c(0.90, 1.0)) +
    labs(
      title = "Specificity Tracking: Recall Rate Control",
      subtitle = "Lower specificity = Higher recall rates = Higher costs",
      x = "Years",
      y = "Specificity (1 - Recall Rate)",
      color = "Performance", 
      shape = "Monthly Decision",
      fill = "Risk Status"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 11, color = "gray60"),
      legend.position = "right",
      panel.grid.minor = element_blank()
    ) +
    annotate("text", x = 6, y = SPECIFICITY_THRESHOLD + 0.005, 
             label = "95% Threshold", color = "red", size = 3) +
    annotate("text", x = 6, y = DUAL_SPECIFICITY + 0.005, 
             label = "Dual Baseline", color = "purple", size = 3)
  
  # Combine plots
  combined_plot <- grid.arrange(p1, p2, ncol = 1, heights = c(1, 1))
  
  return(combined_plot)
}

# ============================================================================
# VISUALIZATION 2: COST-EFFECTIVENESS DASHBOARD
# ============================================================================

create_cost_effectiveness_dashboard <- function(results) {
  # Calculate cumulative costs and decision changes
  results_enhanced <- results %>%
    mutate(
      # Cumulative cost difference (AI - Dual)
      cumulative_cost_diff = cumsum(cost_difference),
      # Decision changes
      decision_change = c(FALSE, diff(decision_numeric) != 0),
      # Annual summaries
      year = ceiling(month / 12)
    ) %>%
    # Calculate annual summaries
    group_by(year) %>%
    mutate(
      annual_ai_months = sum(decision_numeric),
      annual_dual_months = sum(1 - decision_numeric),
      annual_cost_diff = sum(cost_difference)
    ) %>%
    ungroup()
  
  # Plot 1: Monthly Cost Difference
  p1 <- ggplot(results_enhanced, aes(x = month / 12)) +
    geom_col(aes(y = cost_difference / 1000, 
                 fill = ifelse(cost_difference > 0, "AI More Expensive", "AI Less Expensive")),
             alpha = 0.7) +
    geom_hline(yintercept = 0, color = "black", size = 0.5) +
    scale_fill_manual(values = c("AI More Expensive" = "salmon", 
                                 "AI Less Expensive" = "lightblue")) +
    labs(
      title = "Monthly Cost Difference: AI-CAD vs Dual Radiologists",
      subtitle = "Positive values = AI-CAD more expensive",
      x = "Years",
      y = "Cost Difference ($1000s)",
      fill = "Cost Status"
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      panel.grid.minor.x = element_blank()
    )
  
  # Plot 2: Cumulative Cost Impact
  p2 <- ggplot(results_enhanced, aes(x = month / 12)) +
    geom_line(aes(y = cumulative_cost_diff / 1000), 
              color = "darkblue", size = 1.2) +
    geom_area(aes(y = cumulative_cost_diff / 1000), 
              fill = "lightblue", alpha = 0.3) +
    geom_hline(yintercept = 0, color = "black", linetype = "dashed") +
    # Add decision change markers
    geom_vline(data = filter(results_enhanced, decision_change), 
               aes(xintercept = month / 12), 
               color = "red", alpha = 0.5, linetype = "dotted") +
    labs(
      title = "Cumulative Cost Impact Over 7 Years",
      subtitle = "Red lines show decision changes",
      x = "Years",
      y = "Cumulative Cost Difference ($1000s)"
    ) +
    theme_minimal() +
    theme(panel.grid.minor = element_blank())
  
  # Plot 3: Decision Timeline
  p3 <- ggplot(results_enhanced, aes(x = month / 12, y = 1)) +
    geom_tile(aes(fill = decision), height = 0.8, alpha = 0.8) +
    scale_fill_manual(values = c("Continue AI-CAD" = "lightgreen", 
                                 "Switch to Dual" = "lightcoral")) +
    labs(
      title = "Decision Timeline: AI-CAD vs Dual Radiologist",
      x = "Years",
      y = "",
      fill = "Decision"
    ) +
    theme_minimal() +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      panel.grid = element_blank(),
      legend.position = "bottom"
    )
  
  # Plot 4: Risk Alert Timeline
  risk_data <- results_enhanced %>%
    select(month, sensitivity_risk, specificity_risk) %>%
    gather(risk_type, risk_present, -month) %>%
    mutate(
      risk_type = case_when(
        risk_type == "sensitivity_risk" ~ "Sensitivity Risk",
        risk_type == "specificity_risk" ~ "Specificity Risk"
      ),
      year = month / 12
    ) %>%
    filter(risk_present)
  
  p4 <- ggplot(results_enhanced, aes(x = month / 12)) +
    geom_tile(aes(y = "Sensitivity", fill = sensitivity_risk), 
              height = 0.4, alpha = 0.8) +
    geom_tile(aes(y = "Specificity", fill = specificity_risk), 
              height = 0.4, alpha = 0.8) +
    scale_fill_manual(values = c("TRUE" = "orange", "FALSE" = "lightgray")) +
    labs(
      title = "Risk Alert Timeline",
      subtitle = "Orange indicates performance below confidence thresholds",
      x = "Years",
      y = "Risk Type",
      fill = "At Risk"
    ) +
    theme_minimal() +
    theme(
      panel.grid = element_blank(),
      legend.position = "bottom"
    )
  
  # Combine all plots
  dashboard <- grid.arrange(p1, p2, p3, p4, ncol = 2, heights = c(1.2, 1.2, 0.6, 0.6))
  
  return(dashboard)
}

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================

# Create and display visualizations
cat("\n=== Generating Performance Tracking Visualization ===\n")
performance_plot <- create_performance_tracking_plot(simulation_results)

cat("\n=== Generating Cost-Effectiveness Dashboard ===\n") 
dashboard_plot <- create_cost_effectiveness_dashboard(simulation_results)

# Summary statistics for interpretation
cat("\n=== KEY INSIGHTS ===\n")
cat("Final cumulative cost difference: $", 
    round(sum(simulation_results$cost_difference), 0), "\n")

switch_months <- which(simulation_results$decision == "Switch to Dual")
if(length(switch_months) > 0) {
  cat("First switch occurred in month:", min(switch_months), 
      "(Year", ceiling(min(switch_months)/12), ")\n")
  cat("Total months on dual radiologist:", length(switch_months), 
      "out of", N_MONTHS, "\n")
} else {
  cat("Never switched to dual radiologist system\n")
}

# Performance degradation summary
final_sens <- tail(simulation_results$estimated_sensitivity, 1)
final_spec <- tail(simulation_results$estimated_specificity, 1)
cat("Final estimated sensitivity:", round(final_sens, 3), "\n")
cat("Final estimated specificity:", round(final_spec, 3), "\n")

# Risk periods
total_sens_risk_months <- sum(simulation_results$sensitivity_risk)
total_spec_risk_months <- sum(simulation_results$specificity_risk)
cat("Months with sensitivity risk:", total_sens_risk_months, "\n")
cat("Months with specificity risk:", total_spec_risk_months, "\n")
