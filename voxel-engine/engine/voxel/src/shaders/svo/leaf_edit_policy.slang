implementing svo;

public interface ILeafEditPolicy {
    static uint applyLow(uint current, uint command);
    static uint applyHigh(uint current, uint command);
}

public struct OrPolicy : ILeafEditPolicy {
    static uint applyLow(uint current, uint command) {
        return current | command;
    }

    static uint applyHigh(uint current, uint command) {
        return current | command;
    }
}

public struct AndNotPolicy : ILeafEditPolicy {
    static uint applyLow(uint current, uint command) {
        return current & ~command;
    }

    static uint applyHigh(uint current, uint command) {
        return current & ~command;
    }
}
