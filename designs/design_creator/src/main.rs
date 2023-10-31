use std::fs::File;

mod definitions;
use crate::definitions::Design;

fn save(design: &Design, filename: &str) {
    let file = File::create(filename).unwrap();
    serde_json::to_writer_pretty(file, &design).unwrap();
}

macro_rules! save_designs {
    ($($design: ident -> $design_filename: expr),*) => {
        $(mod $design;)*
        $(use crate::$design::$design;)*
        fn main() {
            $(save(&$design(), $design_filename);)*
        }
    };
}
save_designs! {
    short_cantilever -> "../short_cantilever.json",
    cantilever -> "../cantilever.json",
    pipe_bend -> "../pipe_bend.json",
    twin_pipe -> "../twin_pipe.json",
    diffuser -> "../diffuser.json",
    triangle -> "../triangle.json",
    bridge -> "../bridge.json"
}
