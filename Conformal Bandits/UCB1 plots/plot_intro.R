library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)


# Example mapping from method names to μ*
mu_labels <- c(
  "UCB1_0.01" = expression(mu*"*" == 0.01),
  "UCB1_0.05" = expression(mu*"*" == 0.05),
  "UCB1_0.1"  = expression(mu*"*" == 0.1)
)

# Plot 1: Pseudo-regret
p1 <- ggplot(df, aes(x = t, y = mean, color = method, fill = method)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, color = NA) +
  scale_color_manual(values = c("#1b9e77", "#d95f02", "#7570b3"),
                     labels = mu_labels,
                     name = NULL) +   # remove legend title
  scale_fill_manual(values = c("#1b9e77", "#d95f02", "#7570b3"),
                    labels = mu_labels,
                    name = NULL) +   # remove legend title
  labs(x = "Round (t)", y = "Cumulative regret") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top")

# Plot 2: Best arm selection
p2 <- ggplot(df2, aes(x = t, y = mean, color = method, fill = method)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, color = NA) +
  scale_color_manual(values = c("#1b9e77", "#d95f02", "#7570b3"),
                     labels = mu_labels,
                     name = NULL) +   # remove legend title
  scale_fill_manual(values = c("#1b9e77", "#d95f02", "#7570b3"),
                    labels = mu_labels,
                    name = NULL) +   # remove legend title
  labs(x = "Round (t)", y = "Best-arm selection") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top")

# Combine plots side by side with a single legend
combined_plot <- p1 + p2

df3 = three_arms_mu0.05_sd0.1

# --- Histogram plot (p3) without legend, use annotation instead ---
df_arms_long <- df3 %>%
  pivot_longer(cols = everything(), names_to = "Arm", values_to = "Value")

arm_colors <- c(
  "Arm.1" = "#d95f02",
  "Arm.2" = "#f4a582",
  "Arm.3" = "#fddbc7"
)

# --- Compute approximate positions for labels ---
label_data <- df_arms_long %>%
  group_by(Arm) %>%
  summarise(
    x = -0.33,
    y = 3
  ) %>%
  mutate(
    label = case_when(
      Arm == "Arm.1" ~ "mu*\"*\" == 0.05",
      Arm == "Arm.2" ~ "mu == 0.0",
      Arm == "Arm.3" ~ "mu == 0.0"
    )
  )

label_data$y = c(4,3.5,3)
label_data2 = label_data
label_data2$x = label_data2$x - 0.03

# --- Histogram plot (p3) without legend, using geom_text for labels ---
p3 <- ggplot(df_arms_long, aes(x = Value, fill = Arm)) +
  geom_histogram(aes(y = ..density..), position = "identity", alpha = 0.6, bins = 30, color = "black", show.legend = FALSE) +
  scale_fill_manual(values = arm_colors) +
  labs(x = " ", y = "Reward Distribution") +
  xlim(-0.4,0.4) +
  theme_minimal(base_size = 14) +
  # vertical lines
  geom_vline(xintercept = c(0.05, 0, 0), color = arm_colors, size = 0.5) +
  geom_point(data = label_data2, aes(x = x, y = y), color = arm_colors[label_data$Arm], size = 3, shape = 15) + # square
  geom_text(
    data = label_data,
    aes(x = x, y = y, label = label),
    parse = TRUE, hjust = 0, size = 3.5
  ) +
  scale_color_manual(values = arm_colors) +
  theme(legend.position = "none")

p3 <- p3 + guides(fill = "none")  # important: prevents p3 from adding to top legend

# --- Combine the three plots ---
combined_plot2 <- combined_plot + p3 + plot_layout(guides = "collect") & theme(legend.position = "top", legend.justification = c(0.25,2))

combined_plot2

