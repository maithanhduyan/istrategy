

use neat_rust::architecture::group::Group;

#[test]
fn test_group_new() {
    let group = Group::new(5);
    assert_eq!(group.size, 5);
}
