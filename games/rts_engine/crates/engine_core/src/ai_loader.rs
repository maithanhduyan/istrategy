use anyhow::Result;
use bevy::prelude::*;
use libloading::{Library, Symbol};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use shared::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Manager để quản lý việc load/unload AI modules
#[derive(Resource)]
pub struct AiModuleManager {
    loaded_modules: HashMap<String, LoadedAiModule>,
    modules_path: PathBuf,
    watcher_receiver: Option<std::sync::Arc<std::sync::Mutex<Receiver<notify::Result<Event>>>>>,
    _watcher: Option<RecommendedWatcher>,
}

struct LoadedAiModule {
    library: Library,
    metadata: AiModuleMetadata,
    create_fn: Symbol<'static, CreateAiModuleFn>,
    instances: HashMap<u32, Box<dyn UnitAiModule>>, // unit_id -> AI instance
}

impl Default for AiModuleManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AiModuleManager {
    pub fn new() -> Self {
        let modules_path = PathBuf::from("target_modules");

        // Tạo thư mục nếu chưa có
        if !modules_path.exists() {
            std::fs::create_dir_all(&modules_path).unwrap_or_else(|e| {
                error!("Failed to create target_modules directory: {}", e);
            });
        } // Setup file watcher
        let (tx, rx) = channel();
        let mut watcher = RecommendedWatcher::new(tx, notify::Config::default()).ok();

        if let Some(ref mut w) = watcher {
            if let Err(e) = w.watch(&modules_path, RecursiveMode::NonRecursive) {
                error!("Failed to setup file watcher: {}", e);
            }
        }

        Self {
            loaded_modules: HashMap::new(),
            modules_path,
            watcher_receiver: Some(Arc::new(Mutex::new(rx))),
            _watcher: watcher,
        }
    }

    /// Load tất cả AI modules từ thư mục target_modules
    pub fn load_all_modules(&mut self) -> Result<()> {
        if !self.modules_path.exists() {
            warn!("Modules path does not exist: {:?}", self.modules_path);
            return Ok(());
        }

        let entries = std::fs::read_dir(&self.modules_path)?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("so")
                || path.extension().and_then(|s| s.to_str()) == Some("dll")
            {
                let module_name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                if let Err(e) = self.load_module(&module_name, &path) {
                    error!("Failed to load module {}: {}", module_name, e);
                }
            }
        }

        Ok(())
    }

    /// Load một AI module cụ thể
    pub fn load_module(&mut self, name: &str, path: &Path) -> Result<()> {
        info!("Loading AI module: {} from {:?}", name, path);

        // Unload module cũ nếu có
        if self.loaded_modules.contains_key(name) {
            self.unload_module(name);
        }

        unsafe {
            let lib = Library::new(path)?;

            // Load metadata function
            let get_metadata: Symbol<GetModuleMetadataFn> = lib.get(b"get_module_metadata")?;
            let metadata = get_metadata();

            // Load create function
            let create_fn: Symbol<CreateAiModuleFn> = lib.get(b"create_ai_module")?;

            // Store leaked symbols để keep alive
            let create_fn = std::mem::transmute::<
                Symbol<CreateAiModuleFn>,
                Symbol<'static, CreateAiModuleFn>,
            >(create_fn);

            let loaded_module = LoadedAiModule {
                library: lib,
                metadata: metadata.clone(),
                create_fn,
                instances: HashMap::new(),
            };

            self.loaded_modules.insert(name.to_string(), loaded_module);

            info!(
                "Successfully loaded AI module: {} v{}",
                metadata.name, metadata.version
            );
            info!("Supported unit types: {:?}", metadata.supported_unit_types);
        }

        Ok(())
    }

    /// Unload một AI module
    pub fn unload_module(&mut self, name: &str) {
        if let Some(mut module) = self.loaded_modules.remove(name) {
            // Cleanup tất cả instances
            for (unit_id, mut instance) in module.instances.drain() {
                instance.cleanup(unit_id);
            }

            // Library sẽ tự động unload khi drop
            info!("Unloaded AI module: {}", name);
        }
    }

    /// Tạo AI instance cho unit
    pub fn create_ai_instance(&mut self, unit_id: u32, module_name: &str) -> Result<()> {
        let module = self
            .loaded_modules
            .get_mut(module_name)
            .ok_or_else(|| anyhow::anyhow!("Module {} not found", module_name))?;

        unsafe {
            let instance_ptr = (module.create_fn)();
            if instance_ptr.is_null() {
                return Err(anyhow::anyhow!("Failed to create AI instance"));
            }

            let instance = Box::from_raw(instance_ptr);
            module.instances.insert(unit_id, instance);
        }

        info!(
            "Created AI instance for unit {} using module {}",
            unit_id, module_name
        );
        Ok(())
    }

    /// Update AI instance
    pub fn update_ai_instance(
        &mut self,
        unit_id: u32,
        module_name: &str,
        unit_state: &UnitState,
        environment: &EnvironmentInfo,
        delta_time: f32,
    ) -> Option<UnitCommand> {
        let module = self.loaded_modules.get_mut(module_name)?;
        let instance = module.instances.get_mut(&unit_id)?;

        instance.update(unit_state, environment, delta_time)
    }

    /// Initialize AI instance
    pub fn initialize_ai_instance(
        &mut self,
        unit_id: u32,
        module_name: &str,
        unit_state: &UnitState,
    ) {
        if let Some(module) = self.loaded_modules.get_mut(module_name) {
            if let Some(instance) = module.instances.get_mut(&unit_id) {
                instance.initialize(unit_state);
            }
        }
    }

    /// Cleanup AI instance
    pub fn cleanup_ai_instance(&mut self, unit_id: u32, module_name: &str) {
        if let Some(module) = self.loaded_modules.get_mut(module_name) {
            if let Some(mut instance) = module.instances.remove(&unit_id) {
                instance.cleanup(unit_id);
            }
        }
    }

    /// Check for file changes (hot-reload)
    pub fn check_for_changes(&mut self) -> Vec<String> {
        let mut changed_modules = Vec::new();

        if let Some(ref receiver_mutex) = self.watcher_receiver {
            if let Ok(receiver) = receiver_mutex.lock() {
                while let Ok(event_result) = receiver.try_recv() {
                    if let Ok(event) = event_result {
                        match event.kind {
                            EventKind::Create(_) | EventKind::Modify(_) => {
                                for path in event.paths {
                                    if let Some(extension) =
                                        path.extension().and_then(|s| s.to_str())
                                    {
                                        if extension == "so" || extension == "dll" {
                                            if let Some(module_name) =
                                                path.file_stem().and_then(|s| s.to_str())
                                            {
                                                changed_modules.push(module_name.to_string());
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        changed_modules
    }

    /// Reload modules that have changed
    pub fn reload_changed_modules(&mut self, changed_modules: Vec<String>) {
        for module_name in changed_modules {
            let module_path = self.modules_path.join(format!("{}.so", module_name));

            #[cfg(windows)]
            let module_path = self.modules_path.join(format!("{}.dll", module_name));

            if module_path.exists() {
                info!("Hot-reloading module: {}", module_name);

                if let Err(e) = self.load_module(&module_name, &module_path) {
                    error!("Failed to hot-reload module {}: {}", module_name, e);
                }
            }
        }
    }

    /// Get list of loaded modules
    pub fn get_loaded_modules(&self) -> Vec<String> {
        self.loaded_modules.keys().cloned().collect()
    }

    /// Get module metadata
    pub fn get_module_metadata(&self, name: &str) -> Option<&AiModuleMetadata> {
        self.loaded_modules.get(name).map(|m| &m.metadata)
    }
}
