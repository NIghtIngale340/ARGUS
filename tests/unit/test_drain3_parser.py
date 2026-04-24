from pathlib import Path

from src.parsing.drain3_parser import LogParser


def test_load_state_restores_parser_clusters_for_resume(tmp_path: Path) -> None:
    state_path = tmp_path / "drain3_state.bin"
    first_line = (
        "src_user=U1 dst_user=U2 src_computer=C1 dst_computer=C2 "
        "auth_type=Kerberos logon_type=Network auth_orientation=LogOn success=Success"
    )
    second_line = (
        "src_user=U3 dst_user=U4 src_computer=C3 dst_computer=C4 "
        "auth_type=Kerberos logon_type=Network auth_orientation=LogOn success=Success"
    )

    parser = LogParser(state_path=str(state_path))
    first_template_id, _ = parser.parse(first_line)
    second_template_id, _ = parser.parse(second_line)
    template_count = parser.get_template_count()
    parser.save_state()

    resumed_parser = LogParser(state_path=str(state_path))
    assert resumed_parser.load_state()
    assert resumed_parser.get_template_count() == template_count
    assert resumed_parser.parse(first_line)[0] == first_template_id
    assert resumed_parser.parse(second_line)[0] == second_template_id
    assert resumed_parser.get_template_count() == template_count
