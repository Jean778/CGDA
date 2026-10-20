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
# 方法对比：Mixup / CGDA / CG-Mix / CG-Shuffle（budget=0.01）
# ============================================================
suppressMessages({library(ggplot2); library(dplyr); library(tidyr); library(patchwork)})

theme_pub <- theme_classic(base_size = 14) +
  theme(axis.ticks.length = unit(0.18, "cm"),
        axis.ticks         = element_line(linewidth = 0.8, color = "black"),
        axis.text          = element_text(face = "bold", size = 12, color = "black"),
        axis.title         = element_text(face = "bold", size = 14, color = "black"),
        legend.text        = element_text(face = "bold", size = 11),
        legend.title       = element_blank(),
        legend.background  = element_blank(),
        panel.border       = element_blank(),
        axis.line          = element_line(linewidth = 0.8, color = "black"))

d <- read.csv(file.path(REPO, "main_results/panel_correct_protocols_long.csv"))
sc <- read.csv(file.path(REPO, "main_results/sample_counts.csv"))
d <- d %>% filter(budget == 0.01)

method_colors <- c("Mixup" = "#CA704C", "CGDA" = "#258C88",
                   "CG-Mix" = "#7999B9", "CG-Shuffle" = "#F4B183")

make_panel <- function(protocol_use, direction_use) {
  
  d_sub <- d %>% filter(protocol == protocol_use, direction == direction_use)
  
  d_sum <- d_sub %>%
    summarise(
      Mixup = mean(mixup_std) * 100,
      CGDA = mean(cg_sup) * 100,
      `CG-Mix` = mean(cg_mix) * 100,
      `CG-Shuffle` = mean(cg_shuffle) * 100,
      cal_sup = mean(cal_sup) * 100,
      .groups = "drop"
    )
  
  d_sd <- d_sub %>%
    summarise(
      Mixup = sd(mixup_std) * 100,
      CGDA = sd(cg_sup) * 100,
      `CG-Mix` = sd(cg_mix) * 100,
      `CG-Shuffle` = sd(cg_shuffle) * 100,
      .groups = "drop"
    )
  
  d_long <- d_sum %>%
    select(-cal_sup) %>%
    pivot_longer(cols = everything(), names_to = "method", values_to = "mean")
  d_sd_long <- d_sd %>%
    pivot_longer(cols = everything(), names_to = "method", values_to = "sd")
  d_long$sd <- d_sd_long$sd
  
  sc_sub <- sc %>% filter(protocol == protocol_use, direction == direction_use)
  n_cal <- sc_sub$n_cal_b0.1[1]
  
  n_values <- c("Mixup" = 0, "CGDA" = n_cal, "CG-Mix" = 0, "CG-Shuffle" = n_cal)
  d_long$n <- n_values[d_long$method]
  d_long$method <- factor(d_long$method, levels = c("Mixup", "CGDA", "CG-Mix", "CG-Shuffle"))
  
  # 🟥 散点数据（5 seed）
  d_seed <- d_sub %>%
    pivot_longer(cols = c(mixup_std, cg_sup, cg_mix, cg_shuffle),
                 names_to = "method_raw", values_to = "value")
  d_seed$method <- recode(d_seed$method_raw,
                          "mixup_std" = "Mixup", "cg_sup" = "CGDA",
                          "cg_mix" = "CG-Mix", "cg_shuffle" = "CG-Shuffle")
  d_seed$method <- factor(d_seed$method, levels = c("Mixup", "CGDA", "CG-Mix", "CG-Shuffle"))
  d_seed$value <- d_seed$value * 100
  
  # 🟥 CGDA 相对 cal_sup 的平均增益
  delta_cgda <- (d_sum$CGDA - d_sum$cal_sup)
  
  y_max <- max(d_long$mean + d_long$sd)
  y_15 <- y_max * 0.1
  
  ggplot(d_long, aes(x = method, y = mean)) +
    geom_col(aes(color = method), fill = NA, width = 0.4, linewidth = 1.2) +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd),
                  width = 0.2, linewidth = 1.0) +
    # 🟥 浅灰线连接同一种子
    geom_line(data = d_seed, aes(x = method, y = value, group = seed),
              color = "grey70", linewidth = 0.5, alpha = 0.7) +
    # 🟥 散点
    geom_point(data = d_seed, aes(x = method, y = value, color = method),
               size = 2, alpha = 0.9) +
    # 🟥 样本数标注
    geom_text(aes(x = method, y = y_15, label = n),
              size = 4.5, fontface = "bold", color = "black") +
    # 🟥 n= 标注
    annotate("text", x = 0.6, y = y_15,
             label = "n=", size = 4.5, fontface = "bold", color = "black") +
    # 🟥 左上角 direction
    annotate("text", x = 0.5, y = y_max * 0.99,
             label = direction_use, size = 5, fontface = "bold", color = "black", hjust = 0) +
    # 🟥 CGDA 柱子上方标注增益
    annotate("text", x = 2, y = d_sum$CGDA*1.12,
             label = sprintf("Δ = +%.2f pp", delta_cgda),
             size = 4, fontface = "bold", color = "#258C88") +
    scale_color_manual(values = method_colors) +
    scale_y_continuous(limits = c(0, y_max * 1.15), expand = c(0, 0)) +
    labs(x = NULL, y = "Macro-F1 (%)") +
    theme_pub +
    theme(legend.position = "none",
          axis.text.x = element_text(face = "bold", size = 11))
}

p1 <- make_panel("Forward", "8to9")
p2 <- make_panel("LOBO", "1to10")

combined <- p1 + plot_spacer() + p2 + 
  plot_layout(nrow = 3, heights = c(1, 0.07, 1))
combined

# ggsave(file.path(REPO, "main_results/figures/method_comparison.png"), combined,
#        width = 8, height = 10, dpi = 300)
