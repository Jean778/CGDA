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
# CDSL 跨设备漂移：6 气体面板 × 2D PCA
# 排列方式参考数据集 A 紧凑分面；面板内按 3 域着色
# ============================================================
suppressMessages({library(ggplot2); library(ggh4x); library(dplyr)})
d1 <- read.csv(file.path(REPO, "data/CDSL_csv/cdsl_master_slave1.csv"), stringsAsFactors = FALSE)
d2 <- read.csv(file.path(REPO, "data/CDSL_csv/cdsl_master_slave2.csv"), stringsAsFactors = FALSE)
master  <- d1[d1$domain == "master", ]
slave1  <- d1[grepl("slave1", d1$domain), ]
slave2  <- d2[!grepl("master", d2$domain), ]
all <- rbind(master, slave1, slave2)
all$domain <- ifelse(all$domain == "master", "Master",
                     ifelse(grepl("slave1", all$domain), "Slave-1", "Slave-2"))
pca <- prcomp(all[, c("f1","f2","f3","f4","f5","f6")], center = TRUE, scale. = TRUE)
all$PC1 <- pca$x[,1]; all$PC2 <- pca$x[,2]
pct <- round(100*summary(pca)$importance[2,1:2], 1)
# 质心
cents <- all %>% group_by(gas, domain) %>%
  summarise(x = mean(PC1), y = mean(PC2), .groups = "drop")
gas_lab <- c("ammonia"="Ammonia",
             "benzene"="Benzene",
             "carbon monoxide"="Carbon Monoxide",
             "formaldehyde"="Formaldehyde",
             "nitrogen dioxide"="Nitrogen Dioxide",
             "toluene"="Toluene")
all$gas_f <- factor(gas_lab[all$gas], levels = gas_lab)
cents$gas_f <- factor(gas_lab[cents$gas], levels = gas_lab)
xr <- range(all$PC1); yr <- range(all$PC2)
theme_pub <- theme_classic(base_size = 14) +
  theme(axis.ticks.length = unit(0.18, "cm"),
        axis.ticks = element_line(linewidth = 0.8),
        axis.text = element_text(face = "bold", size = 11),
        axis.title = element_text(face = "bold", size = 14),
        legend.text = element_text(face = "bold"),
        legend.title = element_blank(),
        strip.text = element_text(face = "bold", size = 10),
        strip.background = element_rect(fill = "grey98", color = "black", linewidth = 0.7),
        panel.border = element_rect(fill = NA, color = "black", linewidth = 0.7),
        panel.spacing = unit(0.4, "cm"))   # 🟥 0.15 → 0.6

p <- ggplot(all, aes(PC1, PC2)) +
  geom_point(aes(color = domain, shape = domain), size = 1.2, alpha = 0.7) +
  # 质心
  geom_point(data = cents, aes(x, y, color = domain), size = 2.8, shape = 8, stroke = 1.2) +
  # master -> slave1 实线
  geom_segment(data = cents %>% filter(domain == "Master") %>%
                 inner_join(cents %>% filter(domain == "Slave-1"), by = "gas_f",
                            suffix = c("_m", "_s1")),
               aes(x = x_m, y = y_m, xend = x_s1, yend = y_s1),
               color = "grey40", linewidth = 0.6) +
  # master -> slave2 虚线
  geom_segment(data = cents %>% filter(domain == "Master") %>%
                 inner_join(cents %>% filter(domain == "Slave-2"), by = "gas_f",
                            suffix = c("_m", "_s2")),
               aes(x = x_m, y = y_m, xend = x_s2, yend = y_s2),
               color = "grey60", linewidth = 0.6, linetype = "dashed") +
  scale_color_manual(values = c("Master"="black","Slave-1"="#258C88","Slave-2"="#CA704C"),
                     name = NULL) +
  scale_shape_manual(values = c("Master"=16,"Slave-1"=17,"Slave-2"=15), name = NULL) +
  coord_cartesian(xlim = xr, ylim = yr) +
  ggh4x::facet_wrap2(~ gas_f, nrow = 1, axes = "all",
                     labeller = labeller(gas_f = ~ .)) +
  labs(x = paste0("PC1 (", pct[1], "%)"),
       y = paste0("PC2 (", pct[2], "%)")) +
  theme_pub +
  theme(legend.position = "top", legend.justification = "right") +
  guides(color = guide_legend(nrow = 1), shape = guide_legend(nrow = 1))
p
# ggsave(file.path(REPO, "R_figure/cdsl_2d_pca.png"), p, width = 14, height = 7, dpi = 300)
