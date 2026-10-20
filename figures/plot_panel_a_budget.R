# ============================================================
# Repository root (REPO) resolves automatically in Rscript and RStudio.
# Override with env var, e.g.  REPO=<repo-root>  Rscript figures/plot_x.R
# ============================================================
if (Sys.getenv("REPO") != "") {
  REPO <- Sys.getenv("REPO")
} else if (interactive()) {
  if (requireNamespace("rstudioapi", quietly = TRUE)) {
    REPO <- dirname(dirname(rstudioapi::getSourceEditorContext()$path))
  } else {
    REPO <- getwd()
  }
} else {
  .cgda_args <- commandArgs(trailingOnly = FALSE)
  .cgda_file <- sub("^--file=", "", .cgda_args[grepl("^--file=", .cgda_args)])
  REPO <- if (length(.cgda_file) > 0) dirname(dirname(.cgda_file)) else getwd()
}
# ============================================================
# 双面板图：Panel A 散点对比 + Budget 性能曲线（2 行 1 列）
# ============================================================
suppressMessages({library(ggplot2); library(dplyr); library(tidyr); library(patchwork)})

theme_pub <- theme_classic(base_size = 14) +
  theme(axis.ticks.length = unit(0.18, "cm"),
        axis.ticks         = element_line(linewidth = 0.8, color = "black"),
        axis.text          = element_text(face = "bold", size = 12),
        axis.title         = element_text(face = "bold", size = 14),
        legend.text        = element_text(face = "bold", size = 11),
        legend.title       = element_text(face = "bold", size = 11),
        legend.background  = element_blank(),
        panel.border       = element_blank(),
        axis.line          = element_line(linewidth = 0.8, color = "black"))

# ============================================================
# 上图：Panel A（Data A base vs cg_sup 对比散点图）
# ============================================================
d_a <- read.csv(file.path(REPO, "main_results/panel_a_long.csv"))
d_a <- d_a %>% filter(method %in% c("base", "cg_sup"), dataset == "A_UCI_Drift")

seed_cols <- c("7" = "#7999B9", "42" = "#A9D18E", "123" = "#F4B183",
               "2024" = "#6FA9A6", "2025" = "#D83C3C")

d_sum_a <- d_a %>% group_by(method) %>%
  summarise(mean = mean(macro_f1), sd = sd(macro_f1), .groups = "drop")
p_val <- t.test(macro_f1 ~ method, data = d_a)$p.value
sig_lab <- ifelse(p_val < 0.001, "***",
                  ifelse(p_val < 0.01, "**",
                         ifelse(p_val < 0.05, "*", "ns")))

p1 <- ggplot() +
  geom_jitter(data = d_a, aes(x = method, y = macro_f1, color = factor(seed)),
              width = 0.15, size = 2, alpha = 0.7) +
  geom_errorbar(data = d_sum_a, aes(x = method, ymin = mean - sd, ymax = mean + sd),
                width = 0.25, linewidth = 0.8) +
  geom_point(data = d_sum_a, aes(x = method, y = mean),
             size = 3.5, shape = 18) +
  stat_summary(data = d_a, aes(x = method, y = macro_f1, group = seed),
               fun = mean, geom = "line", linewidth = 1, alpha = 0.5) +
  annotate("text", x = 1.5, y = 1.02, label = sig_lab,
           fontface = "bold", size = 6) +
  scale_color_manual(values = seed_cols, name = "Seed") +
  scale_x_discrete(labels = c("base" = "Source Only", "cg_sup" = "CGDA")) +  # 🟥 改成 CGDA
  labs(x = NULL, y = "Macro-F1") +
  theme_pub +
  theme(legend.position = c(0.98, 0.02),
        legend.justification = c(1, 0),
        axis.text.x = element_text(face = "bold", size = 12))

# ============================================================
# 下图：Budget 性能曲线（Cal-Sup / CGDA）
# ============================================================
d_b <- read.csv(file.path(REPO, "main_results/panel_correct_protocols_long.csv"))
d_b <- d_b %>% filter(protocol == "Forward")

d_sum_b <- d_b %>% group_by(budget) %>%
  summarise(cal_sup = mean(cal_sup) * 100,
            cgda = mean(cg_sup) * 100, .groups = "drop")

d_long_b <- d_sum_b %>%
  pivot_longer(cols = c(cal_sup, cgda),
               names_to = "method", values_to = "value")
d_long_b$method <- factor(d_long_b$method, levels = c("cal_sup", "cgda"),
                          labels = c("Cal-Sup", "CGDA"))

p2 <- ggplot(d_long_b, aes(x = budget, y = value, color = method, shape = method)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  scale_color_manual(values = c("Cal-Sup" = "#CA704C", "CGDA" = "#258C88")) +
  scale_shape_manual(values = c("Cal-Sup" = 17, "CGDA" = 15)) +
  scale_x_continuous(breaks = c(0.01, 0.05, 0.10, 0.15, 0.20),
                     labels = c("0.01", "0.05", "0.10", "0.15", "0.20")) +
  labs(x = "Label Budget", y = "Macro-F1 (%)") +
  theme_pub +
  theme(legend.position = c(0.7, 0.2))

# ============================================================
# 2 行 1 列排列
# ============================================================
combined <- p1 + plot_spacer() + p2 + 
  plot_layout(nrow = 3, heights = c(1, 0.07, 1))


combined
combined <- p1 / p2 
combined

# ggsave(file.path(REPO, "main_results/figures/combined_figure.png"), combined,
#        width = 7, height = 10, dpi = 300)
