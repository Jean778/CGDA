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
# 质心对照图：Procrustes 对齐 + 共同 PCA（2×4 面板，图例居中）
# ============================================================
suppressMessages({library(ggplot2); library(dplyr); library(tidyr); library(patchwork)})

theme_pub <- theme_classic(base_size = 14) +
  theme(axis.ticks.length = unit(0.15, "cm"),
        axis.ticks         = element_line(linewidth = 0.8, color = "black"),
        axis.text          = element_text(face = "bold", size = 10, color = "black"),
        axis.title         = element_text(face = "bold", size = 12, color = "black"),
        strip.text         = element_text(face = "bold", size = 11),
        strip.background   = element_rect(fill = "grey92", color = "black", linewidth = 0.7),
        panel.border       = element_rect(fill = NA, color = "black", linewidth = 0.7),
        legend.title       = element_blank())

d <- read.csv(file.path(REPO, "main_results/panel_centroids.csv"))
d <- d %>% filter(budget == 0.01, seed == 42)

gas_cols <- c("#D83C3C", "#7999B9", "#F4B183", "#A9D18E", "#6FA9A6", "#AFABAB")
gas_labels <- c("Acetaldehyde", "Acetone", "Ammonia", "Ethanol", "Ethylene", "Toluene")

methods <- c("base", "cal_sup", "cg_sup", "cg_shuffle")
method_labels <- c("Source Only", "Cal-Sup", "CGDA", "CG-Shuffle")

procrustes_align <- function(src, tgt, ref_src) {
  src_mean <- colMeans(src)
  ref_mean <- colMeans(ref_src)
  src_c <- scale(src, center = TRUE, scale = FALSE)
  ref_c <- scale(ref_src, center = TRUE, scale = FALSE)
  src_scale <- sqrt(mean(src_c^2))
  ref_scale <- sqrt(mean(ref_c^2))
  src_c <- src_c / src_scale
  ref_c <- ref_c / ref_scale
  svd_res <- svd(t(ref_c) %*% src_c)
  R <- svd_res$v %*% t(svd_res$u)
  src_aligned <- src_c %*% R * ref_scale + ref_mean
  tgt_centered <- scale(tgt, center = src_mean, scale = FALSE) / src_scale
  tgt_aligned <- tgt_centered %*% R * ref_scale + ref_mean
  return(list(src = src_aligned, tgt = tgt_aligned))
}

compute_Rc <- function(src_mat, tgt_mat) {
  n_class <- nrow(src_mat)
  rc_vals <- numeric(n_class)
  for (c in 1:n_class) {
    same_dist <- sqrt(sum((tgt_mat[c, ] - src_mat[c, ])^2))
    diff_dists <- sapply(setdiff(1:n_class, c), function(k) {
      sqrt(sum((tgt_mat[c, ] - src_mat[k, ])^2))
    })
    nearest_diff <- min(diff_dists)
    rc_vals[c] <- same_dist / nearest_diff
  }
  return(mean(rc_vals))
}

make_task_panels <- function(protocol_use, direction_use) {
  d_task <- d %>% filter(protocol == protocol_use, direction == direction_use)
  
  cal_src <- d_task %>% filter(method == "cal_sup") %>% arrange(class) %>%
    select(starts_with("src_f")) %>% as.matrix()
  cal_tgt <- d_task %>% filter(method == "cal_sup") %>% arrange(class) %>%
    select(starts_with("tgt_f")) %>% as.matrix()
  
  aligned_data <- list()
  for (m in methods) {
    src <- d_task %>% filter(method == m) %>% arrange(class) %>%
      select(starts_with("src_f")) %>% as.matrix()
    tgt <- d_task %>% filter(method == m) %>% arrange(class) %>%
      select(starts_with("tgt_f")) %>% as.matrix()
    if (m == "cal_sup") {
      aligned_data[[m]] <- list(src = src, tgt = tgt)
    } else {
      aligned <- procrustes_align(src, tgt, cal_src)
      aligned_data[[m]] <- aligned
    }
  }
  
  all_centroids <- do.call(rbind, lapply(methods, function(m) {
    rbind(aligned_data[[m]]$src, aligned_data[[m]]$tgt)
  }))
  colnames(all_centroids) <- paste0("f", 1:ncol(all_centroids))
  pca <- prcomp(all_centroids, center = TRUE, scale. = FALSE)
  pct <- round(100 * summary(pca)$importance[2, 1:2], 1)
  
  plot_data <- list()
  for (m in methods) {
    src_mat <- aligned_data[[m]]$src
    tgt_mat <- aligned_data[[m]]$tgt
    colnames(src_mat) <- paste0("f", 1:ncol(src_mat))
    colnames(tgt_mat) <- paste0("f", 1:ncol(tgt_mat))
    src_pca <- as.data.frame(predict(pca, newdata = src_mat)[, 1:2])
    tgt_pca <- as.data.frame(predict(pca, newdata = tgt_mat)[, 1:2])
    colnames(src_pca) <- c("PC1", "PC2")
    colnames(tgt_pca) <- c("PC1", "PC2")
    plot_data[[m]] <- data.frame(
      PC1 = c(src_pca$PC1, tgt_pca$PC1),
      PC2 = c(src_pca$PC2, tgt_pca$PC2),
      type = rep(c("Source", "Target"), each = 6),
      class = rep(1:6, 2),
      method = m
    )
  }
  
  plot_df <- do.call(rbind, plot_data)
  plot_df$method <- factor(plot_df$method, levels = methods, labels = method_labels)
  plot_df$class <- factor(plot_df$class, levels = 1:6, labels = gas_labels)
  
  r_list <- list()
  for (m in methods) {
    r_list[[m]] <- compute_Rc(aligned_data[[m]]$src, aligned_data[[m]]$tgt)
  }
  
  x_range <- range(plot_df$PC1)
  y_range <- range(plot_df$PC2)
  
  p_list <- list()
  for (i in 1:4) {
    m <- method_labels[i]
    d_sub <- plot_df %>% filter(method == m)
    r_val <- r_list[[methods[i]]]
    
    p_list[[i]] <- ggplot(d_sub, aes(x = PC1, y = PC2, color = class)) +
      geom_segment(data = d_sub %>% group_by(class) %>%
                     filter(type == "Source") %>%
                     left_join(d_sub %>% filter(type == "Target"), by = "class",
                               suffix = c("_src", "_tgt")),
                   aes(x = PC1_src, y = PC2_src, xend = PC1_tgt, yend = PC2_tgt),
                   linewidth = 0.5, alpha = 0.6) +
      geom_point(data = d_sub %>% filter(type == "Source"),
                 shape = 1, size = 3, stroke = 1.2) +
      geom_point(data = d_sub %>% filter(type == "Target"),
                 shape = 17, size = 3) +
      scale_color_manual(values = gas_cols, breaks = gas_labels) +
      labs(title = m,
           x = paste0("PC1 (", pct[1], "%)"),
           y = paste0("PC2 (", pct[2], "%)")) +
      theme_pub +
      theme(legend.position = "none",
            plot.title = element_text(face = "bold", size = 11, hjust = 0.5)) +
      annotate("text", x = -Inf, y = Inf,
               label = sprintf("R̄ = %.2f", r_val),
               hjust = -1.5, vjust = 1.5,
               size = 3.5, fontface = "bold")
  }
  
  return(list(panels = p_list, x_range = x_range, y_range = y_range))
}

# 🟥 两个任务
res_fwd <- make_task_panels("Forward", "8to9")
res_lobo <- make_task_panels("LOBO", "1to10")

# 🟥 全局 x/y 范围
global_x <- range(res_fwd$x_range, res_lobo$x_range)
global_y <- range(res_fwd$y_range, res_lobo$y_range)

for (i in 1:4) {
  res_fwd$panels[[i]] <- res_fwd$panels[[i]] +
    scale_x_continuous(limits = global_x) +
    scale_y_continuous(limits = global_y)
  res_lobo$panels[[i]] <- res_lobo$panels[[i]] +
    scale_x_continuous(limits = global_x) +
    scale_y_continuous(limits = global_y)
}

# 🟥 上面 4 个子图
top_row <- (res_fwd$panels[[1]] | res_fwd$panels[[2]] | res_fwd$panels[[3]] | res_fwd$panels[[4]])

# 🟥 下面 4 个子图
bottom_row <- (res_lobo$panels[[1]] | res_lobo$panels[[2]] | res_lobo$panels[[3]] | res_lobo$panels[[4]])

# 🟥 图例图（1 行排列，放在中间）
legend_plot <- ggplot(data.frame(class = factor(gas_labels, levels = gas_labels)),
                      aes(x = 1, y = 1, color = class)) +
  geom_point(size = 0, alpha = 0) +
  scale_color_manual(values = gas_cols, breaks = gas_labels) +
  theme_void() +
  theme(legend.position = "bottom",
        legend.direction = "horizontal",
        legend.text = element_text(face = "bold", size = 10),
        legend.title = element_blank(),
        plot.margin = margin(0, 0, 0, 0)) +
  guides(color = guide_legend(nrow = 1, override.aes = list(size = 4, alpha = 1)))

# 🟥 3 行排列：上面 4 图 → 图例 → 下面 4 图
combined <- top_row / legend_plot / bottom_row +
  plot_layout(heights = c(10, 0.6, 10))

combined

# ggsave(file.path(REPO, "main_results/figures/centroid_procrustes.png"), combined,
#        width = 14, height = 9, dpi = 300)
