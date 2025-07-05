use clap::{Parser, Subcommand};
use anyhow::{Result, Context};
use std::path::{Path, PathBuf};
use std::process::Command;
use colored::*;

#[derive(Parser)]
#[command(name = "rts_ai_builder")]
#[command(about = "Build tool for RTS Engine AI modules")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build a specific AI module
    Build {
        /// Name of the AI module to build
        module: String,
        /// Build in release mode
        #[arg(short, long)]
        release: bool,
    },
    /// Build all AI modules
    BuildAll {
        /// Build in release mode
        #[arg(short, long)]
        release: bool,
    },
    /// Clean build artifacts
    Clean {
        /// AI module to clean (optional, cleans all if not specified)
        module: Option<String>,
    },
    /// Deploy built modules to target_modules directory
    Deploy {
        /// Specific module to deploy
        module: Option<String>,
    },
    /// Watch for changes and auto-rebuild
    Watch {
        /// Module to watch
        module: String,
    },
    /// List available AI modules
    List,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Build { module, release } => {
            build_module(&module, release)?;
        }
        Commands::BuildAll { release } => {
            build_all_modules(release)?;
        }
        Commands::Clean { module } => {
            clean_modules(module.as_deref())?;
        }
        Commands::Deploy { module } => {
            deploy_modules(module.as_deref())?;
        }
        Commands::Watch { module } => {
            watch_module(&module)?;
        }
        Commands::List => {
            list_modules()?;
        }
    }

    Ok(())
}

fn build_module(module_name: &str, release: bool) -> Result<()> {
    println!("{} Building AI module: {}", "[BUILD]".green().bold(), module_name.cyan());

    let module_path = get_module_path(module_name)?;
    
    let mut cmd = Command::new("cargo");
    cmd.arg("build")
       .current_dir(&module_path);

    if release {
        cmd.arg("--release");
        println!("{} Building in release mode", "[INFO]".blue());
    }

    let output = cmd.output()
        .with_context(|| format!("Failed to run cargo build for {}", module_name))?;

    if output.status.success() {
        println!("{} Successfully built {}", "[SUCCESS]".green().bold(), module_name);
        
        // Auto-deploy after successful build
        deploy_single_module(module_name, release)?;
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("{} Build failed for {}:", "[ERROR]".red().bold(), module_name);
        eprintln!("{}", stderr);
        return Err(anyhow::anyhow!("Build failed"));
    }

    Ok(())
}

fn build_all_modules(release: bool) -> Result<()> {
    println!("{} Building all AI modules", "[BUILD ALL]".green().bold());

    let modules = get_available_modules()?;
    
    for module in modules {
        match build_module(&module, release) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("{} Failed to build {}: {}", "[ERROR]".red().bold(), module, e);
            }
        }
    }

    println!("{} Finished building all modules", "[COMPLETE]".green().bold());
    Ok(())
}

fn clean_modules(module_name: Option<&str>) -> Result<()> {
    match module_name {
        Some(module) => {
            println!("{} Cleaning module: {}", "[CLEAN]".yellow().bold(), module);
            let module_path = get_module_path(module)?;
            
            let output = Command::new("cargo")
                .arg("clean")
                .current_dir(&module_path)
                .output()?;

            if output.status.success() {
                println!("{} Cleaned {}", "[SUCCESS]".green().bold(), module);
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                eprintln!("{} Clean failed: {}", "[ERROR]".red().bold(), stderr);
            }
        }
        None => {
            println!("{} Cleaning all modules", "[CLEAN ALL]".yellow().bold());
            let modules = get_available_modules()?;
            
            for module in modules {
                let _ = clean_modules(Some(&module));
            }
        }
    }

    Ok(())
}

fn deploy_modules(module_name: Option<&str>) -> Result<()> {
    match module_name {
        Some(module) => deploy_single_module(module, false),
        None => {
            println!("{} Deploying all modules", "[DEPLOY ALL]".cyan().bold());
            let modules = get_available_modules()?;
            
            for module in modules {
                let _ = deploy_single_module(&module, false);
            }
            
            Ok(())
        }
    }
}

fn deploy_single_module(module_name: &str, release: bool) -> Result<()> {
    println!("{} Deploying module: {}", "[DEPLOY]".cyan().bold(), module_name);

    let module_path = get_module_path(module_name)?;
    let target_dir = if release { "release" } else { "debug" };
    
    // Determine library extension based on platform
    #[cfg(windows)]
    let lib_extension = "dll";
    #[cfg(not(windows))]
    let lib_extension = "so";
    
    let lib_name = format!("{}.{}", module_name, lib_extension);
    let source_path = module_path.join("target").join(target_dir).join(&lib_name);
    
    // Create target_modules directory if it doesn't exist
    let target_modules_dir = PathBuf::from("target_modules");
    std::fs::create_dir_all(&target_modules_dir)?;
    
    let dest_path = target_modules_dir.join(&lib_name);
    
    if source_path.exists() {
        std::fs::copy(&source_path, &dest_path)
            .with_context(|| format!("Failed to copy {} to target_modules", lib_name))?;
        
        println!("{} Deployed {} to target_modules/", "[SUCCESS]".green().bold(), lib_name);
    } else {
        eprintln!("{} Library not found: {:?}", "[WARNING]".yellow().bold(), source_path);
        eprintln!("         Make sure to build the module first.");
    }

    Ok(())
}

fn watch_module(module_name: &str) -> Result<()> {
    println!("{} Watching module: {} for changes", "[WATCH]".purple().bold(), module_name);
    println!("Press Ctrl+C to stop watching");
    
    // This is a simplified watch implementation
    // In a real implementation, you'd use notify crate for file watching
    loop {
        std::thread::sleep(std::time::Duration::from_secs(2));
        
        // Check if source files have changed (simplified)
        match build_module(module_name, false) {
            Ok(_) => {
                println!("{} Rebuilt {} due to changes", "[REBUILD]".green(), module_name);
            }
            Err(_) => {
                // Ignore build errors in watch mode
            }
        }
    }
}

fn list_modules() -> Result<()> {
    println!("{} Available AI modules:", "[MODULES]".blue().bold());
    
    let modules = get_available_modules()?;
    
    if modules.is_empty() {
        println!("  No AI modules found in ai_modules/ directory");
    } else {
        for module in modules {
            let module_path = get_module_path(&module)?;
            let cargo_toml = module_path.join("Cargo.toml");
            
            if cargo_toml.exists() {
                println!("  {} {}", "✓".green(), module.cyan());
            } else {
                println!("  {} {} (missing Cargo.toml)", "✗".red(), module.yellow());
            }
        }
    }

    Ok(())
}

fn get_available_modules() -> Result<Vec<String>> {
    let ai_modules_dir = PathBuf::from("ai_modules");
    
    if !ai_modules_dir.exists() {
        return Ok(Vec::new());
    }

    let mut modules = Vec::new();
    
    for entry in std::fs::read_dir(ai_modules_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            if let Some(name) = entry.file_name().to_str() {
                modules.push(name.to_string());
            }
        }
    }

    modules.sort();
    Ok(modules)
}

fn get_module_path(module_name: &str) -> Result<PathBuf> {
    let module_path = PathBuf::from("ai_modules").join(module_name);
    
    if !module_path.exists() {
        return Err(anyhow::anyhow!("Module '{}' not found in ai_modules/ directory", module_name));
    }

    Ok(module_path)
}
