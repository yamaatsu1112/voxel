mod game;
mod game_exit;
mod game_init;
mod menu;
mod pause;
mod render_scenes;

pub use game::GameScene;
pub use game_exit::GameExitScene;
pub use game_init::GameInitScene;
pub use menu::MenuScene;
pub use pause::PauseScene;
pub use render_scenes::{RenderGameScene, RenderMenuScene, RenderPauseScene, RenderScenes};

use voxel_engine::{Scene, Scenes};

#[derive(Scenes, Clone, PartialEq)]
pub enum GameScenes {
    Menu(MenuScene),
    GameInit(GameInitScene),
    Game(GameScene),
    GameExit(GameExitScene),
    Pause(PauseScene),
}
