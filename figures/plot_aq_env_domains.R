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
# 数据集 D：环境域划分分析（上下 2 排）
# a = T-RH 平面域划分 | b = 环境域构成的季节迁移
# ============================================================
suppressMessages({library(ggplot2); library(dplyr); library(patchwork)})
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
d <- read.csv(file.path(REPO, "data/raw/AirQualityUCI.csv"), sep = ";", dec = ",",
              header = TRUE, check.names = FALSE, na.strings = c("-200", ""), stringsAsFactors = FALSE)
d <- d[, 1:15]
names(d) <- c("Date", "Time", "CO", "S1_CO", "NMHC", "C6H6", "S2_NMHC",
              "NOx", "S3_NOx", "NO2", "S4_NO2", "S5_O3", "T", "RH", "AH")
d[d == -200] <- NA
d$dt <- as.POSIXct(paste(d$Date, d$Time), format = "%d/%m/%Y %H.%M.%S", tz = "UTC")
d$month <- as.numeric(format(d$dt, "%m"))
d$domain <- ifelse(d$T > 20 & d$RH > 50, "Warm-Humid",
                   ifelse(d$T > 20 & d$RH < 50, "Warm-Dry",
                          ifelse(d$T < 20 & d$RH > 50, "Cold-Humid",
                                 ifelse(d$T < 20 & d$RH < 50, "Cold-Dry", NA))))
d$domain <- factor(d$domain, levels = c("Warm-Humid", "Warm-Dry", "Cold-Humid", "Cold-Dry"))
dom_cols <- c("Warm-Humid" = "#D83C3C", "Warm-Dry" = "#F4B183",
              "Cold-Humid" = "#6FA9A6", "Cold-Dry" = "#7999B9")
# ---- 图 a：T-RH 平面域划分 ----
dom_n <- d %>% filter(!is.na(domain)) %>% count(domain)
n_warm_humid  <- dom_n$n[dom_n$domain == "Warm-Humid"]
n_warm_dry    <- dom_n$n[dom_n$domain == "Warm-Dry"]
n_cold_humid  <- dom_n$n[dom_n$domain == "Cold-Humid"]
n_cold_dry    <- dom_n$n[dom_n$domain == "Cold-Dry"]
pA <- ggplot(d %>% filter(!is.na(domain)), aes(T, RH, color = domain)) +
  geom_point(size = 0.4, alpha = 0.4) +
  geom_vline(xintercept = 20, linetype = "dashed", color = "black", linewidth = 0.8) +
  geom_hline(yintercept = 50, linetype = "dashed", color = "black", linewidth = 0.8) +
  scale_color_manual(values = dom_cols, name = NULL) +
  annotate("text", x = 35, y = 90, label = paste0("n = ", n_warm_humid), fontface = "bold", size = 5) +
  annotate("text", x = 35, y = 15, label = paste0("n = ", n_warm_dry),   fontface = "bold", size = 5) +
  annotate("text", x = 2,  y = 90, label = paste0("n = ", n_cold_humid), fontface = "bold", size = 5) +
  annotate("text", x = 2,  y = 15, label = paste0("n = ", n_cold_dry),   fontface = "bold", size = 5) +
  labs(x = "Temperature (°C)", y = "Relative Humidity (%)") +
  theme_pub +
  theme(legend.position = c(0.62, 0.52), legend.justification = c(0, 0),
        legend.text = element_text(face = "bold", size = 10),
        legend.background = element_blank())
# ---- 图 b：环境域构成的季节迁移（100% 堆叠面积）----
mon_dom <- d %>% filter(!is.na(domain)) %>% count(month, domain) %>%
  group_by(month) %>% mutate(pct = 100 * n / sum(n)) %>% ungroup()
month_n <- mon_dom %>% group_by(month) %>% summarise(total = sum(n))
pB <- ggplot(mon_dom, aes(month, pct, fill = domain)) +
  geom_area(position = "stack", color = "white", linewidth = 0.4) +
  geom_text(data = month_n, aes(x = month, y = 106, label = total),
            inherit.aes = FALSE, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = dom_cols) +
  scale_x_continuous(breaks = 1:12) +
  scale_y_continuous(breaks = seq(0, 100, 25), limits = c(0, 112), expand = c(0, 0)) +
  labs(x = "Month", y = "Proportion of samples (%)") +
  theme_pub +
  theme(legend.position = "none")   # 🟥 图例顶部横排，窄窗不截断
# ---- 上下排列 + 竖尺寸保存 ----
pA / pB
# ggsave(file.path(REPO, "R_figure/D_AB_envdomain.png"),
#        pA / pB, width = 10, height = 13, dpi = 300)
