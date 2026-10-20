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
# 4.8 计算开销帕累托图（A/B/C 三面板，双对数轴）
# ============================================================
suppressMessages({library(ggplot2); library(dplyr); library(ggrepel); library(patchwork)})
theme_pub <- theme_classic(base_size = 14) +
  theme(axis.ticks.length = unit(0.18, "cm"),
        axis.ticks         = element_line(linewidth = 0.8, color = "black"),
        axis.text          = element_text(face = "bold", size = 11, color = "black"),
        axis.title         = element_text(face = "bold", size = 13, color = "black"),
        panel.border       = element_blank(),
        axis.line          = element_line(linewidth = 0.8, color = "black"))
d <- read.csv(file.path(REPO, "main_results/table_46_reproduced_overhead.csv"), stringsAsFactors = FALSE)
d$grp <- ifelse(d$model == "CGDA", "CGDA",
                ifelse(!is.na(d$macro_f1), "Light", "Heavy"))
d$grp <- factor(d$grp, levels = c("CGDA","Light","Heavy"))
d$nx <- 0; d$ny <- 0
d$nx[d$model == "Sniff Multinose"] <- 0.5;  d$ny[d$model == "Sniff Multinose"] <- -0.15
d$nx[d$model == "Sniff ConvNet"]    <- 0.6;  d$ny[d$model == "Sniff ConvNet"]    <- -0.30
d$nx[d$model == "LeNet-5"]          <- -0.4; d$ny[d$model == "LeNet-5"]          <- 0.30
d$nx[d$model == "GasNet"]           <- 0.55; d$ny[d$model == "GasNet"]           <- 0.30
db <- read.csv(file.path(REPO, "main_results/table_cdsl_lit_overhead.csv"), stringsAsFactors = FALSE)
db_avg <- db %>% group_by(model) %>%
  summarise(macro_f1 = mean(macro_f1),
            params_M  = mean(params_M),
            flops_M   = mean(flops_M), .groups = "drop") %>%
  mutate(grp = ifelse(model == "CGDA", "CGDA", "Light"))
mk <- function(dd, lab, show_y = TRUE,
               x_breaks = NULL, y_breaks = NULL,
               x_lim = NULL, y_lim = NULL,
               manual = FALSE) {
  p <- ggplot(dd, aes(x = flops_M, y = params_M)) +
    geom_point(data = dd %>% filter(grp == "Heavy"), color = "grey75", size = 2.2) +
    geom_point(data = dd %>% filter(grp == "Light"), color = "grey45", size = 2.8) +
    geom_point(data = dd %>% filter(grp == "CGDA"), color = "#258C88", size = 4.2, shape = 18)
  if (manual) {
    lab_df <- dd %>% filter(grp == "Light") %>%
      mutate(lbl = sprintf("%s F1=%.3f", model, macro_f1))
    offs <- list(
      "Sniff ConvNet"  = list(dx = 0.03,  dy = 0.0025),
      "Sniff ResNet"   = list(dx = -0.08, dy = 0.004),
      "LSTM"           = list(dx = -0.08, dy = -0.002),
      "LeNet-5"        = list(dx = 0.0,   dy = 0.003)
    )
    lab_df$dx <- sapply(lab_df$model, function(m) offs[[m]]$dx)
    lab_df$dy <- sapply(lab_df$model, function(m) offs[[m]]$dy)
    p <- p +
      geom_segment(data = lab_df,
                   aes(x = flops_M, y = params_M,
                       xend = flops_M + dx, yend = params_M + dy),
                   color = "grey40", linewidth = 0.5) +
      geom_text(data = lab_df,
                aes(x = flops_M + dx, y = params_M + dy, label = lbl),
                size = 2.8, color = "grey25", fontface = "bold", hjust = 0.5) +
      geom_segment(data = dd %>% filter(grp == "CGDA"),
                   aes(x = flops_M, y = params_M,
                       xend = flops_M, yend = params_M + 0.007),
                   color = "#258C88", linewidth = 0.6) +
      geom_text(data = dd %>% filter(grp == "CGDA"),
                aes(x = flops_M, y = params_M + 0.007,
                    label = sprintf("CGDA\nF1 = %.3f", macro_f1)),
                size = 3.4, color = "#258C88", fontface = "bold", lineheight = 0.95,
                hjust = 0.5)
  } else {
    p <- p +
      geom_text_repel(data = dd %>% filter(grp == "Heavy"),
                      aes(label = model), size = 2.6, color = "grey50",
                      max.overlaps = 30, seed = 42) +
      geom_text_repel(data = dd %>% filter(grp == "Light"),
                      aes(label = sprintf("%s F1=%.3f", model, macro_f1)),
                      size = 2.8, color = "grey25",
                      nudge_x = dd$nx[dd$grp == "Light"],
                      nudge_y = dd$ny[dd$grp == "Light"],
                      max.overlaps = 30, seed = 42,
                      box.padding = 0.7, min.segment.length = 0) +
      geom_text_repel(data = dd %>% filter(grp == "CGDA"),
                      aes(label = sprintf("CGDA\nF1 = %.3f", macro_f1)),
                      size = 3.4, color = "#258C88", fontface = "bold",
                      lineheight = 0.95, seed = 42,
                      box.padding = 0.8, min.segment.length = 0)
  }
  sc_x <- scale_x_log10(labels = scales::label_number())
  sc_y <- scale_y_log10(labels = scales::label_number())
  if (!is.null(x_breaks)) sc_x <- scale_x_log10(labels = scales::label_number(),
                                                breaks = x_breaks, limits = x_lim)
  if (!is.null(y_breaks)) sc_y <- scale_y_log10(labels = scales::label_number(),
                                                breaks = y_breaks, limits = y_lim)
  p + sc_x + sc_y +
    labs(title = lab, x = "FLOPs (M)",
         y = if (show_y) "Parameters (M)" else NULL) +
    theme_pub +
    theme(plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
          plot.margin = margin(4, 8, 4, 8))
}
pA <- mk(d %>% filter(input_dim == 128), "Input Dim: 128 (Dataset A)",
         x_breaks = c(0.01, 0.1, 1, 10, 100),
         y_breaks = c(0.01, 0.1, 1, 10))
pB <- mk(db_avg, "Input Dim: 6 (Dataset B)",
         x_breaks = c(0.03, 0.05, 0.1, 0.2, 0.4),
         x_lim = c(0.02, 0.5),
         y_breaks = c(0.01, 0.015, 0.022, 0.035),
         y_lim = c(0.0085, 0.042),
         manual = TRUE)
pC <- mk(d %>% filter(input_dim == 70), "Input Dim: 70 (Dataset C)",
         x_breaks = c(0.01, 0.1, 1, 10, 100),
         y_breaks = c(0.01, 0.1, 1, 10))
fig <- pA | pB | pC
fig
# ggsave(file.path(REPO, "main_results/figures/pareto_46.png"), fig,
#        width = 18, height = 4.8, dpi = 300)
